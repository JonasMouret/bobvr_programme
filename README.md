# BobVr

Automatise le traitement des vidéos GoPro MAX d'une piste de bobsleigh :
on insère une carte SD, et l'application copie, renomme, vérifie, efface,
éjecte, puis convertit la vidéo 360 en équirectangulaire H.264 lisible au doigt
sur un téléphone.

## Le flux complet

1. Une carte est insérée. Elle est reconnue à son **nom de volume** :
   `B1`…`B6` (bobsleigh), `S1`…`S6` (speedluge), `R1`…`R5` (bob racing).
2. Les fichiers `.360` sont copiés depuis `DCIM/100GOPRO` (et, au besoin, la
   racine de la carte) vers la bibliothèque, renommés `B1_1`, `B1_2`, … selon
   la carte et le rang dans la journée.
3. Chaque copie est **relue et comparée par empreinte** à l'original.
4. Les fichiers ne sont effacés de la carte que si la vérification passe. Le
   dossier de la caméra (`DCIM/100GOPRO`) est alors supprimé en entier — ce qui
   emporte aussi les `.LRV` et `.THM` qu'elle laisse derrière — puis **recréé
   vide** au même endroit. Le reste de la carte n'est jamais touché.
5. La carte est éjectée.
6. Chaque clip est converti en équirectangulaire 3840 × 1920, encodé en H.264
   et marqué comme vidéo sphérique.

Si quoi que ce soit échoue, la carte n'est ni vidée ni éjectée.

## Prérequis

```bash
sudo apt install ffmpeg libimage-exiftool-perl
```

* **ffmpeg** doit proposer le filtre `program_opencl` (paquet Ubuntu standard).
* **exiftool** écrit les métadonnées sphériques. Sans lui, la vidéo s'ouvre à
  plat, sans navigation au doigt.
* Un GPU NVIDIA apporte NVDEC, NVENC et OpenCL. Sans GPU, l'application
  fonctionne mais **environ sept fois plus lentement**.

Sous **Windows**, rien de tout cela ne s'installe : on construit un dossier
autonome à double-cliquer — voir [docs/windows.md](docs/windows.md).

Installation :

```bash
pyenv activate BobVr
pip install -e .
```

## Nommer les cartes

C'est à faire **une fois par carte**. L'écriture d'un nom de volume demande des
droits administrateur, donc l'application prépare la commande :

```bash
bobvr-cli label                       # liste les volumes amovibles
bobvr-cli label /dev/sdd1 B1          # indique la marche à suivre
```

Concrètement :

```bash
udisksctl unmount -b /dev/sdd1
sudo fatlabel /dev/sdd1 B1            # ou exfatlabel pour une carte exFAT
```

Retirez puis réinsérez la carte. Un volume dont le nom ne correspond à aucun
engin est **ignoré** — c'est délibéré : l'application efface le contenu des
cartes qu'elle reconnaît.

## Utilisation

```bash
bobvr                                 # interface graphique
bobvr-cli info                        # accélération matérielle détectée
bobvr-cli probe descente.360          # structure d'un fichier
bobvr-cli render descente.360 -o sortie.mp4
bobvr-cli render descente.360 --fov 130     # vue plus large à l'ouverture
bobvr-cli render descente.360 --width 2880  # ou --height 1440, au choix
```

## Champ de vision à l'ouverture

Réglage **Rendu → Vue à l'ouverture → Champ de vision** (80° par défaut, ou
`--fov` en ligne de commande). 80° est la valeur neutre : c'est ce que montrent
VLC et les lecteurs de téléphone, et à cette valeur l'image n'est pas retouchée.
Au-dessus, la vue s'ouvre — 120° montre une fois et demie plus de piste ; en
dessous, elle se resserre.

**Pourquoi c'est appliqué au rendu et pas écrit en métadonnée.** Aucune
spécification de vidéo sphérique ne transporte de champ de vision : ni la v1 de
Google, ni la v2 (`sv3d`). La seule propriété qui existe,
`GPano:InitialHorizontalFOVDegrees`, atterrit dans une boîte XMP racine
qu'aucun lecteur vidéo ne lit. Chaque lecteur choisit donc son propre FOV, et
rien dans le fichier ne peut l'en dissuader.

BobVr le règle donc là où c'est possible : dans la projection.

**La transformation, et pourquoi c'est celle-là.** Un lecteur 360 dessine une
vue rectilinéaire : un point situé à *b* degrés de l'axe atterrit sur l'écran à
`tan(b)`, à l'échelle de son propre champ de vision. Pour que sa fenêtre montre
plus large *à la manière d'un objectif plus large*, la scène doit donc être
écartée dans l'espace des **tangentes**, pas dans celui des angles :

```
tan(angle réel) = c · tan(angle dans le fichier)      c = tan(FOV/2) / tan(40°)
```

La projection du lecteur annule alors exactement les tangentes, et ce qu'il
dessine est une vraie vue rectilinéaire de l'angle voulu : **les lignes droites
restent droites**. Écarter les angles à la place — le réflexe naturel — laisse
une tangente dans la composition et courbe chaque arête : le toit du tunnel
s'arrondit. C'est la différence entre « plus large » et « déformé ».

Le kernel l'écrit `atan2(c·sin α, cos α)`, ce qui tient sur toute la sphère :
la transformation tourne autour de l'axe de visée seulement, laisse 0°, 90° et
180° en place, et reste ordonnée pour tout *c* positif. La sphère est
redistribuée, jamais coupée — rien de noir, rien de déchiré, navigation au
doigt intacte. Ce que l'avant gagne, l'arrière le rend en netteté : plus le FOV
est large, plus le dos de la sphère est grossi et mou. Pour une descente filmée
vers l'avant, c'est le bon compromis, et c'est ce qui fixe la plage à 40–150°.

Sur le repli logiciel (sans GPU), v360 sait projeter la sphère mais pas la
redistribuer, et toutes les approximations possibles avec des filtres simples
courbent les lignes. Le rendu **refuse** donc un FOV autre que 80° sur cette
route, plutôt que de rendre une image tordue.

Le réglage agit au rendu : **les clips déjà convertis doivent être re-rendus**
pour en profiter.

## Vider et réinitialiser la liste

Deux boutons dans la barre d'outils, et un clic droit sur une descente pour la
traiter seule :

* **Vider les terminées** — retire de la liste les descentes déjà converties.
* **Réinitialiser** — vide toute la liste, y compris les descentes en attente
  ou en échec. La numérotation repart de `B1_1`.

Dans les deux cas, une fenêtre demande ce qu'il faut faire des fichiers. Par
défaut **aucun n'est supprimé** : les descentes disparaissent seulement de la
liste. Deux cases permettent d'aller plus loin, avec la place occupée en regard
de chacune :

* supprimer les vidéos converties (`.mp4`) — les originaux restent, tout peut
  être re-rendu ;
* supprimer les fichiers d'origine (`.360`) — **définitif** : plus de
  re-rendu possible, ni changement de champ de vision, ni de résolution, ni
  télémétrie.

Rien n'est supprimé en dehors de la bibliothèque, et une descente qu'un worker
est en train de copier ou de rendre est toujours conservée : la ligne est ce
qui permet de retrouver le fichier en cours d'écriture.

## Organisation des fichiers

```
bibliothèque/
  originaux/2026-02-03/B1_1.360      ← conservé, nécessaire pour re-rendre
  equirect/2026-02-03/B1_1.mp4       ← le rendu
```

## Résolution de sortie

**Rendu → Résolution de sortie** propose des préréglages, et les champs
**Largeur** / **Hauteur** juste en dessous acceptent n'importe quelle taille.
Les deux champs restent liés au format **2:1** : une image équirectangulaire
d'un autre rapport n'est plus une sphère, et les lecteurs la montreraient
déformée. Modifier l'un ajuste l'autre, et le préréglage bascule sur
« Personnalisée ». Les valeurs sont arrondies à ce que H.264 sait encoder
(largeur multiple de 4, hauteur paire).

Au-delà de **4096 px de large**, NVENC en H.264 refuse l'image : le rendu
échoue au lieu de basculer sur le processeur. Les réglages le signalent avant
de lancer une descente pour rien. 4096 × 2048 est donc le maximum utile avec le
GPU ; 3840 × 1920 reste le bon compromis.

En ligne de commande, `--width` ou `--height` suffit — l'autre dimension suit.
Donner les deux avec un rapport différent de 2:1 est refusé plutôt que corrigé
en silence.

## Performances

Mesuré sur la machine de développement (RTX 3080 Ti, Core i5-3570K),
sur une descente de 2 min 42 :

| Pipeline                                       | Vitesse | Durée   |
|------------------------------------------------|---------|---------|
| Implémentation de référence (`geq` + v360, CPU) | 0,08×   | ~34 min |
| Repli logiciel de BobVr                         | 0,17×   | ~16 min |
| **BobVr — kernel OpenCL + NVDEC + NVENC**       | **1,2×** | **~2 min** |

Le gain vient d'un **kernel OpenCL unique** qui effectue toute la
transformation (retrait des recouvrements, mélange des coutures, projection
EAC → équirectangulaire, échantillonnage bicubique) en une seule passe sur le
GPU. Aucune image intermédiaire n'est produite. L'échantillonnage bicubique ne
coûte que 1,3 % de plus que le bilinéaire : le GPU n'est pas le facteur
limitant.

La géométrie de ce kernel a été mesurée sur ffmpeg lui-même, puis validée à
46,2 dB de PSNR contre l'implémentation de référence — voir
[docs/geometry.md](docs/geometry.md).

## Télémétrie

Le décodage GPMF est en place : accéléromètre, gyroscope, magnétomètre,
orientation caméra, gravité, exposition.

```python
from bobvr import telemetry
tel = telemetry.extract(chemin, info, caps)
tel.stream_names()        # ACCL, GYRO, CORI, GRAV, IORI, MAGN, ...
tel.sensor_times("GYRO")  # instant de chaque échantillon, horloge capteur
tel.clock_offset_us       # VPTS - STMP : passage à l'horloge vidéo
tel.speeds_kmh()          # nécessite un point GPS
```

**Point important** : sur la descente analysée, **aucun flux GPS n'est
présent** — la piste est couverte, la caméra n'accroche pas les satellites. Le
tracé de la descente ne pourra donc pas venir du GPS. Les flux `CORI`
(orientation), `GRAV` (gravité) et `ACCL`/`GYRO` sont en revanche bien
enregistrés à 30–800 Hz et permettent des compteurs de G et une stabilisation
de l'horizon. Le dessin des compteurs sur la vidéo reste à faire.

### Temps et repères

Chaque payload porte un `STMP` (horloge capteur) et, à côté des quaternions, un
`VPTS` (horloge vidéo). Leur écart place n'importe quel échantillon sur la
timeline vidéo — la synchronisation se calcule au lieu de se chercher par
corrélation. `sensor_times()` répartit les échantillons dans leur payload ;
« un payload = une seconde » est une approximation, pas une base de calcul.

Les clés `ORIN`/`ORIO` sont décodées et **rapportées, pas appliquées**
(`Stream.axes`, `Stream.axis_plan`, et `remap()` pour les appliquer sciemment).
Mesuré contre `CORI` sur une descente réelle, ce que le fichier déclare —
`XzY` → `ZXY` — ne place pas le gyroscope dans le repère des quaternions ;
c'est `diag(+1, −1, +1)` qui le fait, et `CORI` compose de façon extrinsèque.
Les appliquer d'office reviendrait à ajouter une étape à défaire ensuite.

## Stabilisation

Réglage **Rendu → Stabilisation**, une case à cocher et la force du lissage
(0,5 s par défaut). En ligne de commande :

```bash
bobvr-cli render descente.360 --stabilise 0.5
```

La chaîne est en place : `bobvr/orientation.py` intègre le gyroscope à 802 Hz,
résout le repère de ses axes depuis le fichier lui-même, lisse **à phase
nulle** (filtre appliqué en avant puis en arrière, donc sans retard — ce qu'une
stabilisation embarquée ne peut pas offrir), et le kernel applique une rotation
par image, lue dans une table indexée par le compteur d'images que
`program_opencl` lui passe déjà.

Chaque pièce est vérifiée séparément : l'orientation intégrée colle à `CORI` à
**0,09°** sur une descente de 162 s, une table à l'identité redonne un rendu
identique au bit près, et une table de période 3 confirme que le compteur
compte bien les images.

Décochée par défaut, le temps que le résultat soit jugé sur de vraies
descentes. Les mesures automatiques n'ont pas su trancher — la caméra tourne
d'environ 0,6° par image, soit moins d'un pixel, pendant que le traîneau
avance d'un mètre dans un tunnel : la translation écrase la rotation dans
toute différence d'images. Le contrôle qui a fini par parler est visuel : sur
la courbe la plus appuyée, un verrouillage de la vue déplace le traîneau de
104,9°, exactement la rotation que donne la télémétrie.

En 360 il n'y a pas de budget de recadrage : on tourne la sphère, tous les
pixels existent déjà. Le lissage peut donc être aussi fort qu'on veut — c'est
ce qui rend l'exercice plus simple ici qu'en vidéo plate, pas plus difficile.

Sans kernel OpenCL, le réglage est indisponible : `v360` n'applique qu'une
orientation fixe, pas une par image. Le rendu **refuse** plutôt que de sortir
une vidéo non stabilisée sans le dire.

## Tests

```bash
pytest                                                    # sans vidéo
BOBVR_TEST_360=/chemin/extrait.360 pytest                 # avec vidéo réelle
```

Un extrait court suffit :

```bash
ffmpeg -ss 20 -t 5 -i entree.360 -map 0:0 -map 0:5 -map 0:1 -map 0:3 \
       -c copy -copy_unknown -tag:d gpmd -f mp4 extrait.360
```
