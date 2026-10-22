# Mise en ligne sur Hugging Face Spaces

Le dossier envoyé sur Hugging Face contient uniquement ce qui fait tourner l'application :
`Dockerfile`, `requirements.txt`, `credit_risk/`, `models/` et le `README.md` de ce dossier
(sa première partie, entre les `---`, configure le Space).

Le Dockerfile du Space est celui du projet, avec en plus
`ENV FRAME_ANCESTORS=https://huggingface.co` : Hugging Face affiche l'application dans sa
propre page, ce que l'application interdit par défaut.

Étapes :
1. Sur huggingface.co : New Space, SDK « Docker », modèle « Blank », visibilité « Public ».
2. Onglet « Files » du Space : « Add file » > « Upload files », puis glisser le contenu du dossier.
3. Le Space se construit tout seul (quelques minutes), puis l'application est en ligne.
