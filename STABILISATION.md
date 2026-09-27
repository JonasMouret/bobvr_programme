# Stabilisation — état des lieux

Journal de ce qui a été construit, mesuré, écarté, et de ce qui reste ouvert.
Écrit le 15 août 2026 pour reprendre demain sans refaire le chemin.

**Où on en est en une phrase.** Toute la chaîne est en place, chaque pièce est
vérifiée séparément, et une **cause racine probable a été trouvée en fin de
journée** : la table était construite sur l'horloge des capteurs et indexée par
numéro d'image vidéo, sans jamais appliquer `clock_offset_us` — un décalage
systématique de 38,5 images, soit 1,29 s. C'est corrigé dans
`Renderer._stabilisation()`, mais **le résultat n'est pas encore jugé**.

---

## 1. Ce qui a été construit

### `bobvr/telemetry.py` — décodage GPMF

Existait déjà ; trois corrections apportées :

* **`VPTS` n'est plus traité comme un flux de données.** Il partage le `STRM`
  des quaternions, donc le `SCAL` de 32767 lui était appliqué : un horodatage
  en microsecondes divisé par 32767. Il alimente maintenant `clock_offset_us`.
* **`STMP` est conservé par bloc**, d'où `sensor_times(clé)` qui place chaque
  échantillon sur l'horloge capteur. « Un payload = une seconde » était juste
  à 0,1 % près, faux pour ce qu'on veut en faire.
* **`ORIN`/`ORIO`/`MTRX` sont décodés et rapportés, pas appliqués**
  (`Stream.axes`, `Stream.axis_plan`, fonction `remap()`).

### `bobvr/orientation.py` — nouveau

Python pur, aucune dépendance ajoutée : numpy dans la build Windows autonome
n'en vaut pas la peine, et tous les algorithmes sont linéaires.

| Fonction | Rôle |
|---|---|
| `qmul`, `qconj`, `slerp`, `from_rotvec`… | algèbre des quaternions sur des 4-uplets |
| `solve_gyro_frame(tel)` | résout les axes du gyroscope depuis le fichier ; rend un résidu qui sert d'indice de confiance |
| `integrate(tel, frame)` | intègre les 130 000 échantillons en une orientation |
| `Track.resample(times)` | interpole sur la sphère aux instants voulus |
| `smooth(quats, dt, secondes)` | passe-bas **à phase nulle** (avant puis arrière) |
| `corrections(brut, lissé)` | `raw ⊗ conj(smoothed)` : la rotation par image |
| `stabilise(tel, times, secondes)` | l'enchaînement complet |

Coût sur une descente de 162 s : **1,7 s au total**, négligeable devant un
rendu de 2 min. 18 tests sur télémétrie synthétique à réponse connue.

### `bobvr/render/geometry.py` + le kernel

La table des rotations est **cuite dans la source générée** et indexée par le
compteur d'images que `program_opencl` passe déjà au kernel (`index`,
jusqu'alors ignoré) — c'est la seule chose que ce filtre fait varier dans le
temps.

* trois `int16` par image (seul le vecteur est stocké, `w` étant impliqué par
  le représentant positif) → 6 octets/image ;
* **≈ 6 minutes** tiennent dans les 64 Kio de mémoire constante qu'un GPU doit
  garantir ; au-delà, `TooManyFramesError` plutôt qu'une troncature ;
* `STAB_OFFSET` remet le compteur sur la timeline du clip quand le rendu a
  cherché (`-ss`), pour l'aperçu.

Le bloc `APPLY_STABILISATION` du kernel s'applique **après** `APPLY_ROTATION` :
l'orientation fixe est un choix de cadrage, exprimé dans le repère stabilisé.

### `bobvr/render/pipeline.py`, `config.py`, `ui/settings_dialog.py`, CLI

* `RenderSettings.stabilise_seconds` (0 = désactivé) ;
* `Renderer._stabilisation()` extrait, résout, intègre, lisse — et **refuse**
  si le repère n'est pas fiable (résidu ≥ 20 %) plutôt que de produire une
  correction fausse en silence ;
* le repli logiciel **refuse** aussi : `v360` n'applique qu'une orientation
  fixe, pas une par image ;
* **Rendu → Stabilisation** dans les réglages : une case et la force du
  lissage, grisée quand la case est décochée, tout désactivé sans OpenCL ;
* `bobvr-cli render --stabilise 0.5`.

145 tests passent.

---

## 2. Ce que la télémétrie dit vraiment

Mesuré sur `B21.360` (162 s, 4864 images) — voir `docs/stabilisation/probe_imu.py`.

| Flux | Fréquence | Remarque |
|---|---|---|
| `GYRO` | **802 Hz** | ne sature pas : pic 1 337 °/s pour ±2 000 de pleine échelle |
| `ACCL` | 200 Hz | médiane **2,36 g**, p95 **9,90 g**, max 13,83 g |
| `CORI` | 29,98 Hz | quaternion d'orientation caméra |
| `IORI` | 29,98 Hz | **identité constante** : l'image n'est pas pré-réorientée |
| `GRAV` | 29,98 Hz | vecteur gravité |
| `MAGN` | 25 Hz | 33 à 121 µT — inexploitable |

Trois conclusions qui ont orienté toute la conception :

**Le magnétomètre est à jeter.** Le champ terrestre fait 25–65 µT et devrait
être quasi constant ; ici il varie du simple au quadruple (acier du bob,
ferraillage de la piste). Aucun cap exploitable — et pour stabiliser, un cap
absolu ne sert à rien de toute façon.

**La gravité aussi.** 68 % de la descente au-dessus de 1,3 g : l'accéléromètre
ne mesure pas la verticale, il mesure la résultante en virage. Tout
verrouillage d'horizon basé sur `GRAV`, ou tout Madgwick/Mahony naïf, couche
l'image dans chaque courbe. La verticale doit venir de l'intégration du gyro.

**Le flou de bougé n'est pas un obstacle.** Pose médiane 1/481 s, pire cas
1/240. Au p95 des vitesses de rotation, 0,47° de filé, soit ~5 px en 3840 de
large. L'impeccable est donc physiquement atteignable.

### Répartition spectrale

**30,8 % de la puissance du gyroscope est au-dessus de 15 Hz**, donc
strictement invisible à un flux d'orientation à 30 Hz — dont 14 % entre 60 et
120 Hz. C'est la raison d'utiliser `GYRO` plutôt que `CORI`.

---

## 3. Les conventions, établies par la mesure

Trois faits qui ne sont écrits nulle part dans le fichier et qui ont coûté
cher à retrouver. Voir aussi `docs/stabilisation/curve.py`.

**`CORI` compose de façon extrinsèque.** L'incrément entre deux échantillons
vaut `q(k+1) ⊗ conj(q(k))`, pas l'inverse. Se tromper coûte un résidu de 57 %
qu'aucune permutation d'axes ne rattrape.

**Le gyro brut atteint le repère de `CORI` par `diag(+1, −1, +1)`** — une
simple inversion de signe, sans réordonnancement. Résidu 4,0 %, dérive
**0,09°** sur 162 s.

**Ce n'est pas ce que le fichier déclare.** `ORIN`/`ORIO` annoncent
`XzY → ZXY`, soit `(−r1, r0, r2)`, qui ne place pas le gyroscope dans le
repère des quaternions. D'où le choix de rapporter la déclaration sans
l'appliquer.

**Piège à connaître** : intégrer les taux bruts puis appliquer M au
vecteur-rotation n'est **pas** la même chose qu'appliquer M aux taux puis
intégrer. Elles diffèrent au second ordre — 4 % contre 8 % de résidu. Le
solveur mesure maintenant ce que le pipeline fait réellement.

**Les angles d'Euler sont inutilisables** pour tracer cette séquence : le
tangage passe 60° sur 28 % de la descente, donc lacet et roulis s'échangent
près du blocage de cardan et comptent deux fois la même rotation (~1 270°
chacun). Utiliser des grandeurs indépendantes de la représentation.

---

## 4. Le mécanisme du kernel, vérifié sur GPU

| Contrôle | Résultat |
|---|---|
| Table à l'identité | sortie **identique au bit près** au rendu sans stabilisation |
| Table de lacet croissant | l'image tourne comme prévu (`check_index.py`) |
| **Table de période 3** | images 0 et 3 identiques, 1 et 4 à 11,4 dB, 2 et 5 à 9,8 dB |

Le troisième contrôle est le plus important. `program_opencl` invoque le
kernel **une fois par plan** (Y, U, V) ; un compteur incrémenté par plan
aurait lu la table trois fois trop vite **en passant tous les autres tests**,
puisqu'une rotation qui croît trois fois trop vite croît quand même. Une table
de période 3 sépare les deux cas sans ambiguïté : lue en `k` elle donne
0°, 60°, 120°, 0°… ; lue en `3k` elle donne 0° partout. Le motif observé
confirme que **`index` compte les images**.

---

## 5. Les impasses — ne pas les refaire

### Hypothèses écartées par la mesure

| Hypothèse | Verdict |
|---|---|
| Le lissage | **Non** — verrouiller la vue sur une orientation fixe échoue aussi |
| Le sens de la rotation | **Non** — `q ⊗ conj(s)` et son conjugué testés |
| Un décalage temporel constant | **Non** — balayé ±4 s, courbe plate |
| Une carte d'axes `CORI` → sphère | **Non** — 48 permutations signées, trois critères différents |
| `index` compterait les plans | **Non** — test de période 3 |
| L'image serait pré-réorientée | **Non** — `IORI` est l'identité constante |

### Deux instruments de mesure brûlés

C'est le résultat le plus utile de la journée, et il faut le retenir.

**La différence pixel entre images consécutives ne peut pas voir cet effet.**
La caméra tourne d'environ 0,6° par image, soit **moins d'un pixel**, pendant
que le traîneau avance d'un mètre dans un tunnel. La translation écrase la
rotation, et le moindre artefact de rééchantillonnage dégrade le chiffre — que
la correction soit juste ou fausse. (`check_steady.py`, `find_axes2.py`)

**La corrélation de phase sur la bande équatoriale n'est pas fiable ici.** Sa
courbe de corrélation en fonction du décalage est plate : pic 0,552, moyenne
0,322, aucun pic localisé sur ±120 images. La conclusion « décalage −30
images, donc `VPTS` marque la fin de son payload » qui en avait été tirée est
**fausse et retirée**. (`correlate.py`)

**Pourquoi les recherches ont échoué** : deux inconnues — la carte d'axes et
l'alignement temporel — chacune balayée pendant que l'autre était fausse.

### Le seul contrôle qui ait parlé

Visuel, sur la courbe la plus appuyée (47,7 s, 105° de rotation en 1,5 s) :
en verrouillant la vue, **le traîneau se déplace de −280 px sur 960, soit
−105°** — exactement la rotation que donne la télémétrie sur cette fenêtre —
et le haut reste en haut. C'est la signature d'un comportement correct : le
traîneau étant solidaire de la caméra, c'est lui qui doit bouger quand on
annule la rotation.

---

## 6. La cause racine trouvée en fin de journée

### Le symptôme

Trois rendus complets faits depuis l'interface (`B1_1` à 0,5 s, `B1_2` à
1,5 s, `B1_3` décoché) : **aucune différence perceptible** entre les trois.

Or les images ne sont pas identiques, loin de là. À 47,7 s, comparées à la
version non stabilisée : **13,3 dB** pour 0,5 s, **11,7 dB** pour 1,5 s. La
correction s'applique donc, et plus fort quand on augmente le lissage.

### Le raisonnement

Une correction **décalée dans le temps** produit exactement ce couple de
symptômes. L'orientation de sortie devient `s(k−Δ) ⊗ conj(q(k−Δ)) ⊗ q(k)` : le
terme en `q(k)` laisse passer **tout le tremblement d'origine**, tandis que le
reste recadre globalement la sphère. Résultat : une image aussi tremblante
qu'avant, mais numériquement très différente. C'est mot pour mot ce qui est
observé.

### Le calcul

```
CORI[0] à t_capteur          = 0,1851 s
écart VPTS − STMP            = 1,1011 s
donc CORI[0] à t_vidéo       = 1,2862 s, soit l'image 38,5
```

La table était construite sur l'horloge **capteur** (`sensor_times("CORI")`)
puis indexée par le numéro d'image **vidéo**, sans conversion — alors que
`clock_offset_us` avait été ajouté à `telemetry.py` précisément pour ça.
**Décalage systématique de 38,5 images, soit 1,29 s.**

### Le correctif

`Renderer._stabilisation()` échantillonne maintenant l'orientation aux
instants convertis, une entrée par image vidéo :

```python
fps = info.fps or 30.0
shift = (tel.clock_offset_us or 0) / 1_000_000.0
count = round((info.duration or 0.0) * fps) + 1
times = [k / fps - shift for k in range(count)]
```

### Ce qui reste à vérifier

**Le correctif n'est pas jugé.** La corrélation de phase donne encore « pire »
(0,96 °/image contre 0,46 sans) — mais c'est l'instrument dont la section 5
démontre qu'il est aveugle ici, donc il ne tranche ni dans un sens ni dans
l'autre. Il faut regarder.

Reste aussi ouvert : **`VPTS` marque-t-il le premier ou le dernier échantillon
de son payload ?** Le fichier ne le dit pas, et c'est une ambiguïté d'une
seconde. Si le correctif ci-dessus ne suffit pas, essayer `shift − 1,001`
(une durée de payload) est le deuxième candidat, et il n'y en a que deux.

### Comment juger correctement

Rendre le même extrait avec `stabilise_seconds` à 0 et à 2, ouvrir les deux
**dans un lecteur 360** (pas en équirectangulaire à plat, où 2° ne font que
11 px sur 1920 alors qu'ils occupent une part bien plus grande d'un écran à
80° de champ), et comparer en fixant un point du décor.

Deux fenêtres utiles sur `B21.360` :

* **11–16 s** — la caméra tourne d'environ 16° sur 2 s pendant que le traîneau
  avance encore lentement : la rotation n'est pas noyée dans le défilement des
  parois.
* **45–51 s** — la courbe la plus appuyée, 105° en 1,5 s : si le sens est
  inversé, cela se verra immédiatement.

Le repère qui ne trompe pas : le traîneau est **solidaire de la caméra**. Dans
un rendu correctement stabilisé, c'est donc **le traîneau qui dérive dans le
cadre** pendant que le décor tient. S'il reste immobile pendant que le décor
tourne, la correction ne fait pas son travail.

---

## 7. Après, s'il reste du temps

**L'obturateur déroulant**, les derniers 10 %. Une rotation qui dépend de la
ligne source lue, pas seulement de l'image. Faisable dans ce kernel puisqu'il
calcule déjà la coordonnée source avant d'échantillonner, et les 802 Hz du
gyroscope le permettent — mais il faut mesurer le temps de lecture des
capteurs, qui n'est nulle part dans les métadonnées.

**Le cache des kernels.** Avec une table de stabilisation, la source est
spécifique à un clip : elle est écrite une fois et jamais réutilisée, et rien
ne nettoie le cache aujourd'hui.

---

## 8. Les scripts d'étude

Dans `docs/stabilisation/`. Ils demandent `numpy` (installé dans le
virtualenv, mais pas une dépendance du projet) et se lancent avec le chemin
d'un `.360` en argument.

| Script | Ce qu'il fait |
|---|---|
| `probe_imu.py` | inventaire des flux, fréquences, g, vitesses de rotation |
| `curve.py` | résout le repère, intègre, lisse, exporte les courbes en JSON |
| `check_index.py` | la table est-elle lue par image ? |
| `check_period.py` | `index` compte-t-il les images ou les plans ? **le test à garder** |
| `look.py` | rend trois images d'une fenêtre, avec et sans verrouillage, pour regarder |
| `check_steady.py` | différence pixel entre images — **instrument aveugle**, conservé comme témoin |
| `correlate.py` | corrélation de phase — **instrument non fiable ici** |
| `find_axes2.py`, `find_axes3.py` | recherches de carte d'axes, deux critères |

La page d'analyse publiée à partir de `curve.py` :
<https://claude.ai/code/artifact/18b28f79-641f-4a1e-be83-809644786a45>

---

## 9. Correction du 18 août : la §6 est fausse

**Le décalage d'horloge n'était pas la cause, et le « correctif » cassait un
alignement juste.** Trois mesures indépendantes, toutes faites sur l'image :

| Méthode | Décalage |
|---|---|
| Le lancement : image immobile et gyro immobile qui démarrent ensemble | −0,188 s |
| Ajustement de la rotation mesurée dans l'image contre celle du gyro | −0,1815 s |
| `CORI[0]`, c'est-à-dire « l'échantillon *k* appartient à l'image *k* » | −0,1851 s |

Elles s'accordent à 7 ms. Le code d'avant le 15 août échantillonnait la table
sur `sensor_times("CORI")`, soit −0,1851 s : **il était déjà juste**.
Retrancher `clock_offset_us` (1,1011 s) décale la table de 38 images.

Le repère du lancement est ce qui manquait : c'est un front, pas un motif, donc
l'auto-similarité des parois ne peut pas le brouiller. Sa courbe de corrélation
est piquée — r = 0,918 à l'optimum, 0,52 à ±0,25 s, zéro à 0,8 s — là où les
instruments de la §5 donnaient des courbes plates.

### Un instrument de stabilité qui marche

Rotation globale à 3 degrés de liberté ajustée par flot optique (Lucas-Kanade)
sur la sphère, du grossier au fin sur 4 niveaux, masquée à la bande haute pour
que le traîneau ne pèse pas dans le calcul. **La recherche est bornée par
construction** : elle part de zéro et raffine, donc elle ne peut pas sauter sur
un pic lointain comme le fait une corrélation. C'est exactement ce qui manquait.

Sur la grande courbe (45–51 s) :

| Rendu | Mouvement résiduel |
|---|---|
| sans stabilisation | 3,818 °/img |
| 2 s, ancien alignement | 4,809 °/img |
| 0,5 s, alignement corrigé | 3,867 °/img |
| 2 s, alignement corrigé | **2,495 °/img** |

Le classement est celui que la théorie prédit, y compris le « pire que rien »
du mauvais alignement.

### Ce qui reste

Il manque encore l'essentiel : la télémétrie promet 0,661 → 0,147 °/img, on
n'en obtient que 3,818 → 2,495. **La carte d'axes entre le repère du gyro et
la sphère du kernel reste le suspect.** L'ajustement des deux flux de rotation
donne une matrice quasi diagonale mais de magnitudes 0,15 / 0,37 / 0,63 au lieu
de 1, pour un R² de 0,33 : ce n'est pas une rotation. Soit la carte est fausse,
soit l'instrument est encore trop bruité pour la lire. C'est la prochaine
question, et elle est maintenant isolée — l'alignement temporel ne la
contamine plus.

### Correction à la §5

« 0,6°/image fait moins d'un pixel » est faux d'un facteur 7 : un pixel couvre
0,094° à 3840 de large. Les deux instruments restent écartés, mais pour la
bonne raison — les parois du tunnel se ressemblent trop, et la corrélation sort
−142° en une seconde là où le gyro est calme (12 % de mesures aberrantes).

### La carte d'axes est l'identité — le suspect est éliminé

Les 48 permutations signées classées en notant le cosinus entre la rotation
mesurée dans l'image et `A · ω_gyro`. Le vainqueur est `−x−y−z` (0,688) et le
dernier `+x+y+z` (−0,688) : ce n'est qu'un signe global, celui de la convention
de l'estimateur. Les six permutations suivantes sont toutes sous 0,58. **Les
repères correspondent déjà** ; il n'y a pas de remapping à faire, et la §5
peut fermer cette piste pour de bon.

### Les limites de l'instrument, à connaître avant de s'en servir

Mesuré contre le gyro sur 45–65 s : pentes **0,31 / 0,83 / 0,57** selon l'axe,
corrélations 0,39 / 0,52 / 0,59. Autrement dit il ne restitue que la moitié de
la rotation, et son plancher est de **1,05 °/img** sur les sections rapides
(flot de translation) et 0,34 °/img sur les sections lentes (bruit).

Conséquence directe : **il sait constater un échec, pas certifier un succès.**
Il a bien détecté le mauvais alignement (4,809 contre 3,818 sans rien), mais la
télémétrie ne promet que 0,80 °/img sur la grande courbe — c'est-à-dire sous
son plancher. Élargir le cône ou itérer plus dégrade le gain au lieu de
l'améliorer (0,13/0,22/0,27 à 90°).

Mesures sur 45–51 s, cône et masque du traîneau recalés dans le repère caméra :

| Rendu | Mesuré | Promis par la télémétrie |
|---|---|---|
| sans stabilisation | 1,616 | 1,977 |
| 0,5 s | 1,295 | 1,353 |
| 2 s | 1,546 | 0,796 |

Le brut colle à la prédiction une fois le gain de 0,82 appliqué, le 0,5 s aussi,
le 2 s montre un excès inexpliqué. Deux hypothèses, aucune testée :
l'**obturateur déroulant** (à 9 °/img de pointe et ~15 ms de lecture, 4° de
cisaillement intra-image, que rien de global ne peut annuler — ce serait un
plancher, pas un défaut de la correction), et un biais de l'estimateur qui
croît avec l'amplitude de la correction.

**Pour trancher, il faut un meilleur instrument**, pas un meilleur réglage de
celui-ci : suivi de points avec RANSAC (OpenCV n'est pas installé dans le
virtualenv, c'est le seul obstacle), qui donnerait un gain de 1 au lieu de 0,5.

---

## 10. Le 18 août, seconde moitié : l'obturateur déroulant, mesuré

Avec OpenCV installé dans le virtualenv, l'instrument devient exact : suivi de
points + RANSAC sur la sphère (`docs/stabilisation/track_rotation.py`). Sur des
rotations connues appliquées à une vraie image il rend **1,503 pour 1,5°** et
**5,996 pour 6°**, à 0,003° près, et sa répétabilité sur la vidéo est de
0,000 °/img (p90 0,226). Ce n'est plus un instrument, c'est une mesure.

### Le gyro ne décrit pas le rapide

| Bande | R² image contre gyro |
|---|---|
| lent (> 0,5 s) | +0,287 |
| moyen (> 0,17 s) | +0,448 |
| **rapide** | **+0,047** |

Le gyro suit l'image dans le lent et pas du tout dans le rapide — c'est-à-dire
exactement la bande que la stabilisation existe pour enlever. Comme
l'estimateur est déterministe, ce n'est pas du bruit.

### La cause, et sa valeur

Le capteur ne photographie pas un instant : il lit ligne après ligne. L'image
porte donc l'orientation **moyennée sur la durée de lecture**. En modélisant
ça et en balayant la durée T :

| T | R² total | R² rapide |
|---|---|---|
| 0 (obturateur global) | 0,382 | 0,148 |
| 0,5 image | 0,476 | 0,419 |
| **1,0 image = 33 ms** | **0,503** | **0,497** |
| 1,5 image | 0,503 | 0,480 |
| 2,0 images | 0,482 | 0,396 |

**T ≈ 1 période image, 33 ms.** C'est le chiffre que la §7 disait introuvable
dans les métadonnées. Il explique aussi le « +0,5 image » trouvé plus tôt : une
moyenne sur [t, t+1 image] est centrée sur t+0,5.

La table était construite sur l'orientation *instantanée* : on sur-corrigeait
précisément dans la bande visée. `orientation.stabilise()` échantillonne
maintenant l'orientation moyennée (`blur()`, `READOUT_FRAMES`).

### Le gain, mesuré

Grande courbe, 45–51 s, cône et masque du traîneau recalés dans le repère
caméra :

| Rendu | médiane | p75 |
|---|---|---|
| sans stabilisation | 1,623 | 2,785 |
| 0,5 s — ancienne table | 1,205 | 2,251 |
| **0,5 s — avec lecture capteur** | **1,077** | **1,743** |
| 2 s — ancienne table | 1,359 | 1,998 |
| **2 s — avec lecture capteur** | **1,084** | **1,539** |

Soit −33 % en médiane et −45 % au p75 contre le brut, et −20 à −23 % apporté
par la seule prise en compte de la lecture.

### Ce qui reste, et ce n'est plus un mystère

La télémétrie promet 1,986 → 0,797 à 2 s. Le traqueur sous-lit d'environ 18 %
(brut mesuré 1,623 pour 1,986 promis), donc 0,5 s tombe pile sur sa prédiction
(1,077 mesuré, 1,12 attendu) — mais 2 s plafonne à 1,084 au lieu de 0,65.

**Ce plancher est le cisaillement intra-image**, celui que `blur()` ne peut pas
corriger : les lignes viennent d'instants différents, et aucune rotation par
image ne défait ça. C'est exactement le chantier de la §7, sauf qu'il a
maintenant son paramètre mesuré (33 ms) et sa taille connue (~1,08 °/img sur la
courbe la plus dure). Augmenter le lissage au-delà de 0,5 s ne sert plus à rien
tant qu'il n'est pas fait.

---

## 11. La correction par ligne : faite, mesurée, et laissée désactivée

### Ce qui est en place

Le kernel sait maintenant faire varier la rotation *à l'intérieur* d'une image.
`to_source()` a été extraite en fonction (le calcul doit être fait deux fois :
une pour savoir d'où vient l'échantillon, une une fois l'instant connu), et une
petite algèbre de quaternions a été ajoutée — `qmul`, `qconj`, `qrot`, `qpow`
pour la rotation fractionnaire.

**La vitesse ne coûte aucune mémoire** : elle se lit dans la table elle-même,
en différence centrée sur deux images. L'erreur que ça introduit vaut le
déplacement de la trajectoire *lissée* entre voisins — moins d'un dixième de
degré, sous ce qu'on corrige. Le budget de 64 Kio et les 6 minutes tiennent.

`RenderSettings.rolling_axis` / `rolling_span` pilotent le tout. Contrôle passé :
avec `rolling_axis=None`, la sortie est **identique au bit près** à avant.

### La géométrie du balayage, mesurée

Gradient temporel ajusté face par face, en séparant les moitiés de chaque face
et en régressant l'écart de rotation contre ω. Trois faces ont assez de points :

| Face | Piste | Axe | dt (images) | r |
|---|---|---|---|---|
| FRONT | 0 | `fb` | −0,565 | −0,335 |
| RIGHT | 0 | `fb` | −0,397 | −0,321 |
| BACK | 1 | `fa` | +0,398 | +0,473 |

`fa` de BACK vaut `1−b` : les trois gradients pointent donc dans la **même**
direction, l'axe vertical propre à chaque face, dans les deux pistes. L'instant
ne dépend donc pas de l'empaquetage EAC mais de la seule **élévation** — d'où
`ROLLING_AXES` exprimé dans la direction et non dans les coordonnées source.

### Pourquoi c'est resté désactivé

**Le gain n'a pas pu être démontré.** Cisaillement résiduel mesuré (écart de
rotation haut/bas régressé sur ω) : brut +0,721 ; par image seulement +0,512 ;
`up-last` à 1,0/1,5/2,5 → +0,547/+0,543/+0,599 ; `up-first` +0,446. Aucun ordre
n'améliore franchement, et le signe qui aide un peu (`up-first`) **contredit**
celui que donne l'ajustement par face (`up-last`).

Cette contradiction est la signature d'une mesure confondue, et la cause est
identifiable : sur une descente, la vitesse et le braquage varient ensemble, si
bien que la **parallaxe de translation** — qui suit la vitesse linéaire — imite
un cisaillement proportionnel à ω. La régression ne les sépare pas.

**Et la théorie dit pourquoi l'effet est faible.** Un décalage temporel entre
lignes donne, d'une image à l'autre, un écart proportionnel à l'accélération
angulaire, pas à ω : au premier ordre la déformation est la même dans les deux
images et s'annule. Ici |dω| vaut 0,628 °/img² en médiane, donc ~0,6 °/img de
gauchissement **non rigide** — réel, mais invisible à toute mesure qui ajuste
une rotation globale. C'est un défaut de *fidélité géométrique*, pas de
stabilité.

### Le résultat qui compte

Le plancher du traqueur, mesuré là où le gyro est calme (0,19–0,63 °/img), vaut
**0,83 °/img** : c'est de la parallaxe, pas du tremblement. Or le rendu à 2 s
mesure 1,084, soit un résidu réel de √(1,084² − 0,83²) ≈ **0,70 °/img** contre
**0,797 promis** par la télémétrie.

**La stabilisation atteint donc déjà sa cible.** Le « plancher » attribué en
§10 à l'obturateur déroulant était celui de l'instrument. La §10 est corrigée
sur ce point : monter le lissage au-delà de 0,5 s sert bien à quelque chose.

Ce qui reste pour la correction par ligne est un travail de *qualité d'image*
(lignes droites qui le restent en virage appuyé), pas de stabilité, et il lui
faut une mesure qui sépare la parallaxe : filmer une séquence en rotation sans
translation, ou régresser contre l'accélération angulaire plutôt que contre ω.
