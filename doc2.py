semantic_roles = {}

for token in docs:
    if token.dep_ != "ROOT":
        # Using the entire subtree for each token to capture phrases
        subtree_span = docs[token.left_edge.i : token.right_edge.i + 1]
        role_info = {
            "text": subtree_span.text,  # The whole phrase
            "dep": token.dep_,  # Dependency relation to the head
            "pos": token.pos_,  # Part of Speech tag
            "token": token,  # The token itself
            "tag": token.tag_,  # Fine-grained part of speech tag
            "index": token.i,  # Index of the token in the sentence
        }

        head = token.head
        if head not in semantic_roles:
            semantic_roles[head] = []
        semantic_roles[head].append(role_info)

for head, roles in semantic_roles.items():
    print(
        f"Head: {head.text} (POS: {head.pos_}) (dep: {head.dep_}) (tag: {head.tag_}) (index: {head.i})"
    )

    # Sort roles based on dependency relation

    roles.append(
        {
            "text": head.text,
            "dep": head.dep_,
            "pos": head.pos_,
            "token": head,
            "tag": head.tag_,
            "index": head.i,
        }
    )
    sorted_roles = sorted(roles, key=lambda role: role["token"].i)

    composite = " ".join([role["text"] for role in sorted_roles])

    print(f"  Composite: {composite}")

    for role in sorted_roles:
        print(f"  Role: {role}")

# Initialize structure to hold semantic roles
semantic_rolesb = {
    "who": [],
    "did": [],
    "whom/what": [],
    "when": [],
    "where": [],
    "why": [],
    "how": [],
}

# Mapping of dependency tags to semantic roles
dep_to_role = {
    "nsubj": "who",
    "csubj": "who",
    "csubjpass": "who",
    "nsubjpass": "who",  # Subjects
    # "ROOT": "did",  # Verbs
    # "aux": "did",
    # "xcomp": "did",
    "ccomp": "did",
    "advcl": "did",
    "dobj": "whom/what",
    "pobj": "whom/what",  # Objects
    "acomp": "how",
    "oprd": "how",  # Manner
    "pcomp": "where",  # Place
    "npadvmod": "when",
    "advmod": "when",  # Time
    "relcl": "did",  # Relative clauses
}

# Iterate over tokens to assign roles
actions = []
facts = []
root_verb = None

for token in docs:
    role = (
        dep_to_role.get(token.dep_, None)
        if token.dep_ != "ROOT"
        else (
            "did"
            if token.tag_ in ["VERB", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]
            else "who"
        )
    )

    if token.dep_ == "conj" and token.tag_ in [
        "VERB",
        "VB",
        "VBD",
        "VBG",
        "VBN",
        "VBP",
        "VBZ",
    ]:
        role = "did"

    if token.dep_ == "ROOT":
        root_verb = token
    if token.dep_ == "ROOT" and token.pos_ == "AUX":
        if token not in semantic_rolesb["did"]:
            semantic_rolesb["did"].append(token)

    # if token.dep_ == "nsubj" and token.head.dep_ == "ROOT":
    #     semantic_rolesb["did"].append(token.text)
    if token.head.dep_ in ["aux", "auxpass"]:
        head = token.head
        rel_pos = token.i - head.i
        # Find 'head' in one of the lists from semantic_rolesb values
        if rel_pos != -1:
            for key, value in semantic_rolesb.items():
                if head.text in value:
                    # Change the value of the key to the new value
                    semantic_rolesb[key].remove(head)

    if role and role not in ["who", "whom/what"]:
        if token.i not in [tk.i for tk in semantic_rolesb[role]]:
            semantic_rolesb[role].append(token)

    elif role in ["who", "whom/what"]:
        for noun_chunk in noun_phrases:
            if token.text in noun_chunk.text:
                semantic_rolesb[role].append(noun_chunk.root)
    elif token.dep_ == "prep" and token.head.dep_ in [
        "ROOT",
        "advcl",
    ]:  # Handle prepositions for why/where
        for child in token.children:
            if child.dep_ == "pobj":
                semantic_rolesb["where" if token.head.dep_ == "ROOT" else "why"].append(
                    child
                )
    if role == "how":

        if token.dep_ == "acomp":
            for child in token.children:
                if child.dep_ == "prep":
                    semantic_rolesb["how"].append(child)

    if role == "when":
        if token.dep_ == "npadvmod":
            for child in token.children:
                if child.dep_ == "prep":
                    semantic_rolesb["where"].append(child)
        if token.dep_ == "advmod":
            for child in token.children:
                if child.dep_ == "prep":
                    semantic_rolesb["why"].append(child)
    if role == "where":
        if token.dep_ == "pcomp":
            for child in token.children:
                if child.dep_ == "prep":
                    semantic_rolesb["where"].append(child)
        if token.dep_ == "advmod":
            for child in token.children:
                if child.dep_ == "prep":
                    semantic_rolesb["where"].append(child)

    # Find main actions and subjects
    if token.dep_ == "ROOT" or (
        token.pos_ == "VERB" and "subj" in {child.dep_ for child in token.children}
    ):
        actiona = " ".join(
            [child.text for child in token.subtree if child.dep_ != "punct"]
        )
        actions.append(actiona)

    # Extract facts about specific subjects, considering nested and complex sentences
    if (
        token.pos_ == "NOUN" or token.pos_ == "PROPN" or token.pos_ == "PRON"
    ) and token.head.pos_ in [
        "VERB",
        "VB",
        "VBD",
        "VBG",
        "VBN",
        "VBP",
        "VBZ",
    ]:
        rv = self.find_root_verb(token.head)

        if rv:
            subject_actions = " ".join(
                [rv.text]
                + [
                    child.text
                    for child in rv.children
                    if (child.dep_ != "nsubj" or child.dep_ != "nsubjpass")
                    and child.pos_ != "PUNCT"
                ]
            )
            facts.append((token.text, subject_actions))
        facts.extend([(token.text, fact) for fact in self.extract_facts(token)])
    # if role == "why":
    #     if token.dep_ == "advcl":
    #         for child in token.children:
    #             if child.dep_ == "prep":
    #                 semantic_rolesb["why"].append(child.text)

# print(f"Actions: {actions}")
# print(f"Facts: {facts}")
print(semantic_rolesb)
# Check if the number of subordinate clauses is equal to the number of actions minus 1
assert (
    num_subordinate_clauses == len(semantic_rolesb["did"]) - 1
    if num_subordinate_clauses > 0
    else True
)

# check if the ROOT verb has an auxiliary verb
subj_compound = []
obj_compound = []
action_compound = []
compound = []
dobj_text = ""
templates = []
for action in semantic_rolesb["did"]:
    # if action.dep_ == "advcl":
    #     print(f"Action: {action.text}")
    #     exit(0)
    # Find the subject of the action in the original docs object
    for token in docs:
        if token.text == action.text and token.dep_ == action.dep_:
            subj_compound = []
            subj = [
                tk
                for tk in token.children
                if tk.dep_ == "nsubj" or tk.dep_ == "nsubjpass"
            ]
            if len(subj) == 0:
                subj = [
                    tk
                    for tk in token.head.children
                    if tk.dep_ == "nsubj" or tk.dep_ == "nsubjpass"
                ]
            obj_compound = []
            subj_text = subj[0].text if subj else ""
            compound = (
                [
                    tk
                    for tk in subj[0].children
                    if tk.dep_ == "compound"
                    or tk.dep_ == "poss"
                    or tk.dep_ == "amod"
                    or tk.dep_ == "det"
                    or tk.dep_ == "aux"
                ]
                if subj
                else []
            )
            if (
                token.dep_ == "conj"
                and tok.pos_ in ["NOUN", "PROPN"]
                and compound == []
            ):
                for sbj in token.head.children:
                    if sbj.dep_ == "cc" and sbj.head == token.head:
                        direction_from_conj = "left" if tok.i > sbj.i else "right"

                        connected_word = (
                            docs[sbj.i - 1]
                            if direction_from_conj == "left"
                            else docs[sbj.i + 1]
                        )
                        while connected_word.dep_ in [
                            "compound",
                            "amod",
                            "poss",
                            "det",
                            "aux",
                        ]:
                            connected_word = (
                                docs[connected_word.i - 1]
                                if direction_from_conj == "left"
                                else docs[connected_word.i + 1]
                            )
                        if connected_word.dep_ in ["dobj", "pobj"]:
                            break
                        if connected_word.dep_ in [
                            "nsubj",
                            "nsubjpass",
                            "csubj",
                            "csubjpass",
                        ]:
                            subj.append(connected_word)
                            compound.append(connected_word)
                            compound.append(sbj)
                            if token not in subj:
                                subj.append(token)
                            if token not in compound:
                                compound.append(token)

            elif (
                token.dep_ == "conj"
                and token.pos_
                in [
                    "VERB",
                    "VB",
                    "VBD",
                    "VBG",
                    "VBN",
                    "VBP",
                    "VBZ",
                ]
                and len(subj) == 0
                and len(compound) == 0
            ):
                if token.head.dep_ == "ROOT":

                    cw_subject = [
                        tk
                        for tk in token.head.children
                        if tk.dep_ == "nsubj" or tk.dep_ == "nsubjpass"
                    ]
                    if len(cw_subject) == 0:
                        cw_subject = [
                            tk
                            for tk in token.head.head.children
                            if tk.dep_ == "nsubj" or tk.dep_ == "nsubjpass"
                        ]
                    print(f"cw_subject: {cw_subject}")
                    compound.extend(cw_subject)
                    subj.extend(cw_subject)

                    for tk in cw_subject:
                        for tkk in tk.children:
                            if tkk.dep_ in [
                                "compound",
                                "poss",
                                "amod",
                                "det",
                                "aux",
                            ]:
                                compound.append(tkk)
                        for tkk in tk.head.children:
                            if tkk.dep_ in [
                                "compound",
                                "poss",
                                "amod",
                                "det",
                                "aux",
                            ]:
                                compound.append(tkk)
                else:
                    assert root_verb is not None
                    cw_subject = [
                        tk
                        for tk in root_verb.children
                        if tk.dep_ == "nsubj" or tk.dep_ == "nsubjpass"
                    ]
                    if len(cw_subject) == 0:
                        cw_subject = [
                            tk
                            for tk in root_verb.head.children
                            if tk.dep_ == "nsubj" or tk.dep_ == "nsubjpass"
                        ]
                    print(f"cw_subject: {cw_subject}")
                    compound.extend(cw_subject)
                    if len(subj) == 0:
                        subj.extend(cw_subject)

                    for tk in cw_subject:
                        for tkk in tk.children:
                            if tkk.dep_ in [
                                "compound",
                                "poss",
                                "amod",
                                "det",
                                "aux",
                            ]:
                                if tkk not in compound:
                                    compound.append(tkk)
                        for tkk in tk.head.children:
                            if (
                                tkk.dep_
                                in [
                                    "compound",
                                    "poss",
                                    "amod",
                                    "det",
                                    "aux",
                                    "acomp",
                                    "advmod",
                                    "npadvmod",
                                    "attr",
                                ]
                                and token.i < tkk.i
                            ):
                                # print(f"FART1: {tkk}")
                                # exit(0)
                                if tkk not in compound:
                                    if (
                                        tkk.dep_ != "acomp"
                                        or (
                                            tkk.dep_ == "acomp"
                                            and (
                                                token.tag_ == "VBP"
                                                or token.tag_ == "VBZ"
                                            )
                                            and token == tkk.head
                                        )
                                        and tkk.dep_ != "npadvmod"
                                    ):
                                        # compound.append(tkk)
                                        # subj_compound.append(tkk)
                                        if tkk not in obj_compound:
                                            obj_compound.append(tkk)
                                        for tkki in tkk.children:
                                            if tkki.dep_ in [
                                                "compound",
                                                "poss",
                                                "amod",
                                                "det",
                                                "aux",
                                            ]:
                                                if tkki not in obj_compound:
                                                    # compound.append(tkki)
                                                    obj_compound.append(tkki)
                                if (tkk.dep_ == "npadvmod" or tkk.dep_ == "attr") and (
                                    token == tkk.head
                                    if tkk.head.pos_ == "VERB"
                                    else (True if token == tkk.head.head else False)
                                ):

                                    if tkk not in obj_compound:
                                        obj_compound.append(tkk)
                                    if tkk.dep_ == "attr":
                                        for tkki in tkk.children:
                                            if tkki not in obj_compound:
                                                obj_compound.append(tkki)
                                    for tkki in tkk.children:
                                        if tkki.dep_ in [
                                            "compound",
                                            "poss",
                                            "amod",
                                            "det",
                                            "aux",
                                        ]:
                                            if tkki not in obj_compound:
                                                obj_compound.append(tkki)
                                        elif "WP" or "WDT" in [
                                            tki.tag_ for tki in tkk.children
                                        ]:
                                            for tki in tkk.children:
                                                if (
                                                    tki.tag_ == "WP"
                                                    or tki.tag_ == "WDT"
                                                ):
                                                    if tki not in subj_compound:
                                                        subj_compound.append(tki)
                                                    for tkki in tki.head.children:

                                                        if (
                                                            tkki not in subj_compound
                                                            and tkk.dep_ != "npadvmod"
                                                        ):
                                                            subj_compound.append(tkki)
            else:
                if len(subj) == 0:
                    print(f"\n Why is subj empty? {subj} {token.text} \n")
                    subj = [
                        tk
                        for tk in root_verb.children
                        if tk.dep_ == "nsubj" or tk.dep_ == "nsubjpass"
                    ]
