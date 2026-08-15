# Géométrie du .360 : comment les constantes ont été obtenues

Ce document explique d'où viennent les nombres qui figurent dans
`bobvr/render/geometry.py` et dans le kernel OpenCL. Ils n'ont pas été
recopiés d'un autre projet : ils ont été mesurés, puis vérifiés.

## Ce que contient un fichier .360

Un fichier GoPro MAX `.360` est un MP4 qui contient **deux pistes vidéo HEVC**
de 4096 × 1344 (une par objectif), une piste audio AAC, une piste de timecode
et une piste `gpmd` de télémétrie.

Ensemble, les deux pistes portent les six faces d'un **cubemap équi-angulaire**
(EAC). Mais ce n'est pas une simple grille : les deux objectifs se recouvrent,
et chaque piste enfouit une couture de raccord **au milieu** de sa première et
de sa troisième face.

```
piste 0 :  [ GAUCHE |couture|  ][    AVANT     ][ DROITE |couture|  ]
piste 1 :  [  BAS   |couture|  ][   ARRIÈRE    ][  HAUT  |couture|  ]
```

Ce placement a un sens physique : l'objectif avant couvre entièrement la face
AVANT et la moitié des faces GAUCHE et DROITE. La couture tombe donc là où les
deux objectifs se rejoignent, pas sur une frontière de face.

## Mesure de la projection EAC → équirectangulaire

Plutôt que de réimplémenter de mémoire les mathématiques EAC de ffmpeg (source
d'erreurs difficiles à détecter), la correspondance exacte a été **extraite de
ffmpeg lui-même** :

1. Construire une image 4032 × 2688 en `rgb48le` où chaque pixel encode ses
   propres coordonnées (canal R = x, canal G = y).
2. La faire passer par `v360=eac:e:w=3840:h=1920:interp=nearest`.
3. Lire la sortie : chaque pixel porte désormais les coordonnées entières du
   pixel source dont il provient.

On obtient ainsi la vérité terrain de ffmpeg, sans supposition.

### Disposition des faces

En comparant cette carte à un calcul analytique de `xyz_to_cube`, la
correspondance direction → case de la grille 3 × 2 ressort avec une **pureté de
1,0000** (aucun pixel mal classé) :

| Face   | Case (ligne, colonne) | Orientation stockée |
|--------|-----------------------|---------------------|
| GAUCHE | (0, 0)                | miroir horizontal   |
| AVANT  | (0, 1)                | aucune              |
| DROITE | (0, 2)                | miroir horizontal   |
| BAS    | (1, 0)                | rotation 270°       |
| ARRIÈRE| (1, 1)                | rotation 90°        |
| HAUT   | (1, 2)                | rotation 270°       |

Chaque orientation gagne contre la deuxième meilleure candidate avec un écart
massif : 1 à 2 px d'erreur moyenne contre 485 à 647 px.

### Vérification sub-pixel

L'erreur résiduelle a ensuite été décomposée. En x, l'écart moyen vaut
exactement **+0,500 px avec σ = 0,289** — soit précisément la signature d'un
arrondi `floor()` sur une distribution uniforme (σ = 1/√12 = 0,289). La
géométrie est donc exacte au pixel près.

Une régression linéaire donne les formules réellement appliquées par ffmpeg :

```
px = 2 + (a + colonne) × 1342,667      (marge globale de 2 px)
py = 2 + b × 1340 + ligne × 1344       (marge de 2 px par face)
```

Ces marges de 2 px (`pixel_pad` dans le code de ffmpeg) évitent d'échantillonner
au-delà du bord d'une face lorsqu'on travaille sur une image EAC déjà
assemblée. **BobVr ne les reproduit pas** : le kernel échantillonne directement
les pistes brutes, où les vrais pixels de recouvrement existent de part et
d'autre des coutures. Chaque face conserve donc sa taille exacte au lieu d'être
imperceptiblement rétrécie.

### Validation du kernel

Le kernel a été comparé image par image à l'implémentation de référence
(`geq` + `v360` sur CPU) :

| Configuration                                   | PSNR      | SSIM  |
|-------------------------------------------------|-----------|-------|
| Kernel BobVr (sans marge)                       | 35,2 dB   | 0,942 |
| Kernel BobVr **avec émulation de la marge**     | **46,2 dB** | **0,989** |

Le bond de 35 à 46 dB dès que la marge est émulée démontre que **la totalité de
l'écart provenait de cette marge**, et non d'une erreur de projection.

## Constantes de couture

Pour une piste de 4096 × 1344 :

| Constante        | Valeur | Origine                                    |
|------------------|--------|--------------------------------------------|
| face             | 1344   | hauteur de la piste (face carrée)          |
| largeur EAC      | 4032   | 3 × face                                   |
| perte par couture| 32     | (4096 − 4032) / 2                          |
| demi-couture     | 64     | largeur de mélange propre à GoPro          |
| couture en source| 128    | 2 × demi-couture                           |
| couture en EAC   | 96     | 128 − 32                                   |
| couture 0 (EAC)  | 624    | centrée dans la face 0 : (1344 − 96) / 2   |
| couture 1 (EAC)  | 3312   | 2 × 1344 + 624                             |
| couture 0 (source)| 624   | —                                          |
| couture 1 (source)| 3344  | 3312 + 32                                  |

Le mélange reproduit celui de GoPro : sur les 64 pixels de la couture, le pixel
`X` du premier objectif est fondu avec le pixel `X + 64` du second, avec un
poids `(X + 1) / 65`. C'est le seul endroit où les deux objectifs sont
réconciliés ; le sauter laisse une ligne visible sur toute la hauteur de
l'image.

## Format de sortie

`v360` produit par défaut du 4032 × **2389** pour une entrée EAC 4032 × 2688 —
un rapport de 1,687:1. Or une vidéo équirectangulaire **doit être en 2:1** pour
être interprétée correctement par les lecteurs. L'implémentation de référence
recadre en 4032 × 2388, ce qui conserve ce rapport erroné et produit une image
déformée une fois marquée comme sphérique.

BobVr impose explicitement un rapport 2:1 (3840 × 1920 par défaut) et refuse
toute autre proportion.
