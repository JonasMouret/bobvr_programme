# Construire BobVr pour Windows

Le résultat est un dossier `dist\BobVr\` autonome : on le copie sur le poste de
la piste, on double-clique `BobVr.exe`, et c'est tout. Rien à installer côté
poste — pas de Python, pas de PATH à régler.

## Sur la machine de build, une fois

1. **Python 3.11 ou plus récent**, 64 bits, depuis
   [python.org](https://www.python.org/downloads/windows/). À l'installation,
   cochez **« Add python.exe to PATH »**.
2. Le projet, copié ou cloné quelque part sans espaces exotiques dans le
   chemin.

Rien d'autre : pas de Visual Studio, pas de SDK. PySide6 et PyInstaller
arrivent en roues précompilées.

## Les outils externes

BobVr appelle deux programmes qu'il n'embarque pas lui-même. Déposez-les dans
`installer\vendor\` **avant** de construire, et ils voyageront dans la build :

```
installer\vendor\
    ffmpeg.exe
    ffprobe.exe
```

Les métadonnées 360 (ce qui fait qu'un lecteur ouvre la vidéo en sphère plutôt
qu'à plat) sont écrites par BobVr lui-même, en Python : exiftool n'est plus
nécessaire.

### ffmpeg / ffprobe

**Attention à la version.** N'importe quel ffmpeg ne fait pas l'affaire : il
faut une build qui contienne à la fois `program_opencl` (le kernel de
projection) et `h264_nvenc` (l'encodeur GPU). Sans OpenCL, BobVr tombe sur le
repli logiciel, environ sept fois plus lent, et le réglage de champ de vision
devient indisponible.

Deux sources habituelles fournissent des builds complètes :

* [gyan.dev](https://www.gyan.dev/ffmpeg/builds/) — prendre
  **ffmpeg-release-full**, pas *essentials* ;
* [BtbN/FFmpeg-Builds](https://github.com/BtbN/FFmpeg-Builds/releases) —
  prendre une **win64-gpl**.

Vérifiez la build avant de la déposer, plutôt que de vous fier au nom :

```powershell
.\ffmpeg.exe -hide_banner -filters   | findstr program_opencl
.\ffmpeg.exe -hide_banner -encoders  | findstr h264_nvenc
```

Les deux lignes doivent répondre. Une fois BobVr construit, `bobvr-cli.exe
info` redit la même chose en français, avec ce qui manque.

### Le GPU

NVDEC, NVENC et OpenCL viennent du **pilote NVIDIA** du poste, pas de la build.
Un pilote GeForce à jour suffit. Sur une machine sans GPU NVIDIA, BobVr
fonctionne mais bascule sur le repli logiciel.

## Construire

Depuis la racine du projet, dans PowerShell :

```powershell
.\installer\build_windows.ps1
```

Le script crée son propre environnement dans `.venv-build`, installe tout,
lance PyInstaller, vérifie que les fichiers indispensables sont bien là — dont
le kernel OpenCL — et affiche pour finir l'accélération détectée.

Ajoutez `-Zip` pour obtenir en plus `dist\BobVr-0.1.0-win64.zip`, prêt à
copier.

Si PowerShell refuse d'exécuter le script :

```powershell
powershell -ExecutionPolicy Bypass -File .\installer\build_windows.ps1
```

Pour appeler PyInstaller directement :

```powershell
pyinstaller installer\bobvr.spec --noconfirm --clean
```

## Ce que contient le résultat

```
dist\BobVr\
    BobVr.exe          ← l'interface, à double-cliquer
    bobvr-cli.exe      ← la ligne de commande (info, probe, render, label)
    _internal\         ← Qt, Python, le kernel OpenCL, et vendor\ (ffmpeg, ffprobe)
```

Environ 220 Mo, dont l'essentiel est Qt. Le dossier se déplace d'un bloc :
`BobVr.exe` cherche ses outils à côté de lui, dans `_internal\vendor\`, puis sur
le PATH — dans cet ordre. (PyInstaller range les fichiers embarqués sous
`_internal\` ; BobVr sait les y trouver.)

## Premier démarrage sur le poste de la piste

1. Copiez le dossier `BobVr` où vous voulez, par exemple `C:\BobVr`.
2. Lancez `bobvr-cli.exe info`. Il doit annoncer OpenCL et NVENC. Si non,
   relisez la section ffmpeg ci-dessus — c'est presque toujours la build.
3. Lancez `BobVr.exe`, ouvrez **Réglages** et vérifiez le dossier de la
   bibliothèque (par défaut `Vidéos\BobVr`).
4. **Nommez les cartes**, une fois par carte. Sous Windows, c'est la commande
   `label` du système :

   ```powershell
   .\bobvr-cli.exe label                 # liste les cartes insérées
   .\bobvr-cli.exe label E: B1           # nomme la carte du lecteur E:
   ```

   L'écriture du nom demande une invite de commandes **administrateur** ;
   sinon BobVr affiche la commande exacte à coller. Retirez puis réinsérez la
   carte ensuite.

Une carte dont le nom ne correspond à aucun engin est ignorée — c'est
délibéré : BobVr efface le contenu des cartes qu'il reconnaît.

## Signature et SmartScreen

L'exécutable n'est pas signé. Au premier lancement, Windows affichera
« Windows a protégé votre ordinateur » : *Informations complémentaires* →
*Exécuter quand même*. Pour éviter cet écran sur un poste partagé, il faut un
certificat de signature de code, qui s'achète — hors sujet ici, mais c'est la
seule solution propre.

L'antivirus peut aussi regarder de travers un exécutable PyInstaller fraîchement
construit : c'est un faux positif classique, sans rapport avec le contenu.

## Ce qui n'a pas été vérifié sur Windows

La build a été validée sur Linux : PyInstaller produit bien les deux
exécutables, le kernel OpenCL est embarqué au bon endroit et un rendu complet
passe depuis l'exécutable gelé. Le fichier `.spec` est le même sur les deux
systèmes.

Restent à confirmer sur le poste Windows lui-même, faute d'un Windows sous la
main ici :

* la détection et l'éjection des cartes (`bobvr/cards/windows.py`, écrit
  d'après la documentation Win32, jamais exécuté) ;
* l'écriture du nom de volume par `label` ;
* le comportement de l'OpenCL selon le pilote NVIDIA installé.

Les trois se testent en cinq minutes avec une carte dans le lecteur, et
`bobvr-cli.exe info` puis `label` disent précisément ce qui manque.
