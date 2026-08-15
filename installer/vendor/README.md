# vendor

Déposez ici les exécutables à embarquer dans la build Windows :

```
vendor/
  ffmpeg.exe
  ffprobe.exe
  exiftool.exe
  exiftool_files/        (dossier livré avec exiftool.exe, à copier entier)
```

Tout ce que contient ce dossier est copié dans `dist/BobVr/vendor/`, où
l'application va les chercher toute seule — aucun réglage, aucun PATH.

Laissez-le vide et l'application cherchera `ffmpeg`, `ffprobe` et `exiftool`
sur le PATH du poste, ou à côté de `BobVr.exe`.

Voir `docs/windows.md` pour où les télécharger et quelle version prendre.
