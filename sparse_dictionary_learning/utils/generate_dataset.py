import pandas as pd
import random

dataset = 2

if dataset == 1:
    vocab = {
        "animal_terrestre": ["le lion", "le chat", "l'éléphant", "le chien", "le cerf"],
        "animal_aquatique": ["le dauphin", "le requin", "la baleine", "le poulpe", "la truite"],
        "plante_arbre": ["le chêne", "le sapin", "le baobab", "le palmier", "l'érable"],
        "plante_fleur": ["la rose", "la tulipe", "la marguerite", "le tournesol", "l'orchidée"],
        "objet_alimentaire": ["la pomme", "le pain", "le fromage", "le gâteau", "la carotte"],
        "objet_non_alimentaire": ["le marteau", "le stylo", "la chaise", "le téléphone", "la clé"]
    }

    verbs_by_subject = {
        "animal": ["observe", "évite", "cherche", "ignore", "contourne"],
        "plante": ["côtoie", "abrite", "surplombe", "décore"],
        "objet": ["touche", "encombre", "accompagne", "remplace"]
    }

    mapping = {
        "animal_terrestre": "animal", "animal_aquatique": "animal",
        "plante_arbre": "plante", "plante_fleur": "plante",
        "objet_alimentaire": "objet", "objet_non_alimentaire": "objet"
    }

    def generate_coherent_dataset(n_samples=1000):
        data = []
        categories = list(vocab.keys())

        for _ in range(n_samples):
            cat_sujet = random.choice(categories)
            cat_objet = random.choice(categories)

            type_sujet = mapping[cat_sujet]

            sujet = random.choice(vocab[cat_sujet])
            verbe = random.choice(verbs_by_subject[type_sujet])
            objet = random.choice(vocab[cat_objet])

            sentence = f"{sujet.capitalize()} {verbe} {objet}."

            row = {
                "sentence": sentence,
                "animal": 1 if mapping[cat_sujet] == "animal" or mapping[cat_objet] == "animal" else 0,
                "objet": 1 if mapping[cat_sujet] == "objet" or mapping[cat_objet] == "objet" else 0,
                "plante": 1 if mapping[cat_sujet] == "plante" or mapping[cat_objet] == "plante" else 0,
                "animal_terrestre": 1 if "terrestre" in cat_sujet or "terrestre" in cat_objet else 0,
                "animal_aquatique": 1 if "aquatique" in cat_sujet or "aquatique" in cat_objet else 0,
                "plante_arbre": 1 if "arbre" in cat_sujet or "arbre" in cat_objet else 0,
                "plante_fleur": 1 if "fleur" in cat_sujet or "fleur" in cat_objet else 0,
                "objet_alimentaire": 1 if (cat_sujet == "objet_alimentaire" or cat_objet == "objet_alimentaire") else 0,
                "objet_non_alimentaire": 1 if "non_alimentaire" in cat_sujet or "non_alimentaire" in cat_objet else 0
            }
            data.append(row)

        return pd.DataFrame(data)

    df = generate_coherent_dataset(1000)
    df.to_csv("/Users/maxime/MSV_Brain/sparse_dictionary_learning/data/categorial_concepts.csv", index=False, encoding="utf-8")

    print(df["sentence"].head(10))

elif dataset == 2:
    import random

    taxonomy = {
        "animal": {
            "terrestre": ["le lion", "le chat", "le chien", "le cerf", "l'éléphant"],
            "aquatique": ["le dauphin", "le requin", "la baleine", "le poulpe", "la truite"]
        },
        "plante": {
            "arbre": ["le chêne", "le sapin", "le baobab", "le palmier", "l'érable"],
            "fleur": ["la rose", "la tulipe", "la marguerite", "le tournesol", "l'orchidée"]
        },
        "objet": {
            "alimentaire": ["la pomme", "le pain", "le fromage", "le gâteau", "la carotte"],
            "outil": ["le marteau", "le stylo", "la pelle", "le tournevis", "la hache"]
        }
    }

    verbs = {
        "vivant": ["observe", "évite", "contourne", "découvre", "frôle"],
        "non_vivant": ["touche", "encombre", "accompagne", "heurte", "longe"]
    }

    def generate_hierarchical_dataset(n_samples=1200):
        rows = []

        specific_cats = [
            ("animal", "terrestre"), ("animal", "aquatique"),
            ("plante", "arbre"), ("plante", "fleur"),
            ("objet", "alimentaire"), ("objet", "outil")
        ]

        for _ in range(n_samples):
            # Sélection du sujet et de l'objet
            (type_s, spec_s) = random.choice(specific_cats)
            (type_o, spec_o) = random.choice(specific_cats)

            sujet = random.choice(taxonomy[type_s][spec_s])
            objet = random.choice(taxonomy[type_o][spec_o])

            # Sélection du verbe selon si le sujet est animé ou non
            action_type = "vivant" if type_s in ["animal", "plante"] else "non_vivant"
            verbe = random.choice(verbs[action_type])

            sentence = f"{sujet.capitalize()} {verbe} {objet}."

            # Initialisation des labels à 0
            label = {
                "sentence": sentence,
                "etre_vivant": 0, "objet": 0,
                "animal": 0, "plante": 0,
                "terrestre": 0, "aquatique": 0, "arbre": 0, "fleur": 0,
                "alimentaire": 0, "outil": 0
            }

            for cat in [(type_s, spec_s), (type_o, spec_o)]:
                t, s = cat
                #  1
                if t in ["animal", "plante"]: label["etre_vivant"] = 1
                else: label["objet"] = 1

                #  2
                if t in label: label[t] = 1

                #  3
                if s in label: label[s] = 1

            rows.append(label)

        return pd.DataFrame(rows)

    df = generate_hierarchical_dataset(1500)
    df.to_csv("/Users/maxime/MSV_Brain/sparse_dictionary_learning/data/categorial_concepts_2.csv", index=False, encoding="utf-8")

    print(f"Dataset généré : {len(df)} phrases.")
    print("Exemple :\n", df[["sentence", "etre_vivant", "animal", "terrestre"]].head(3))