# vendor

Déposez ici les exécutables à embarquer dans la build Windows :

```
vendor/
  ffmpeg.exe
  ffprobe.exe
```

Tout ce que contient ce dossier est copié dans `dist/BobVr/_internal/vendor/`,
où l'application va les chercher toute seule — aucun réglage, aucun PATH.

Laissez-le vide et l'application cherchera `ffmpeg` et `ffprobe` sur le PATH du
poste, ou à côté de `BobVr.exe`.

Les métadonnées 360 sont écrites par BobVr en Python : exiftool n'est plus
nécessaire.

Voir `docs/windows.md` pour où télécharger ffmpeg et quelle version prendre.
