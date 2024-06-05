import re
import spacy


def main(docs, nlp: spacy.lang.en.English):
    # Initialize structure to hold semantic roles
    # if docs is a list of strings, convert it to a spacy object using the nlp object
    if isinstance(docs, list):
        docs = nlp(docs)
    semantic_rolesb = {
        "who": [],
        "did": [],
        "whom/what": [],
        "when": [],
        "where": [],
        "why": [],
        "how": [],
    }
    noun_phrases = [chunk for chunk in docs.noun_chunks]
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
                    semantic_rolesb[
                        "where" if token.head.dep_ == "ROOT" else "why"
                    ].append(child)
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
    # assert (
    #     num_subordinate_clauses == len(semantic_rolesb["did"]) - 1
    #     if num_subordinate_clauses > 0
    #     else True
    # )

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
                                    if (
                                        tkk.dep_ == "npadvmod" or tkk.dep_ == "attr"
                                    ) and (
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
                                                                tkki
                                                                not in subj_compound
                                                                and tkk.dep_
                                                                != "npadvmod"
                                                            ):
                                                                subj_compound.append(
                                                                    tkki
                                                                )
                else:
                    if len(subj) == 0:
                        print(f"\n Why is subj empty? {subj} {token.text} \n")
                        subj = [
                            tk
                            for tk in root_verb.children
                            if tk.dep_ == "nsubj" or tk.dep_ == "nsubjpass"
                        ]
                    for tkk in subj[0].head.children:
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
                            ]
                            and token.i < tkk.i
                        ):
                            # print(f"FART2: {tkk}")
                            # exit(0)
                            if tkk not in compound:
                                if (
                                    tkk.dep_ != "acomp"
                                    or (
                                        tkk.dep_ == "acomp"
                                        and (token.tag_ == "VBP" or token.tag_ == "VBZ")
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
                                            "npadvmod",
                                        ]:
                                            if tkki not in obj_compound:
                                                # compound.append(tkki)
                                                obj_compound.append(tkki)

                            if tkk.dep_ == "npadvmod" and token == tkk.head:

                                if tkk not in obj_compound:
                                    obj_compound.append(tkk)
                                for tkki in tkk.children:
                                    if tkki.dep_ in [
                                        "compound",
                                        "poss",
                                        "amod",
                                        "det",
                                        "aux",
                                        "npadvmod",
                                    ]:
                                        if tkki not in obj_compound:
                                            obj_compound.append(tkki)
                                print(f"obj_compound npadvmod: {obj_compound}")

                        elif tkk.tag_ == "WP" or tkk.tag_ == "WDT":
                            if tkk not in subj_compound:
                                subj_compound.append(tkk)
                            for tki in tkk.children:

                                if tki not in subj_compound:
                                    subj_compound.append(tki)
                        elif "WP" or "WDT" in [tki.tag_ for tki in tkk.children]:
                            for tki in tkk.children:
                                if tki.tag_ == "WP" or tki.tag_ == "WDT":
                                    if (
                                        tki not in subj_compound
                                        and tkk.dep_ != "npadvmod"
                                    ):
                                        subj_compound.append(tki)
                                    for tkki in tki.head.children:

                                        if (
                                            tkki not in subj_compound
                                            and tkk.dep_ != "npadvmod"
                                        ):
                                            subj_compound.append(tkki)
                    print(f"subj_compound: {subj_compound}")

                subj_text = subj[0].text if subj else ""
                print(f"Compound: {compound}")
                # if compound and len(compound) > 0:
                #     obj_compound.extend(compound)

                for tk in subj[0].children:

                    if len(compound) > 0:
                        if len(subj_compound) == 0:
                            subj_compound = [subj[0]]
                        for tkk in tk.head.children:
                            if (
                                tkk.dep_
                                in [
                                    "compound",
                                    "poss",
                                    "amod",
                                    "det",
                                    "aux",
                                ]
                                and tkk not in subj_compound
                                and tkk not in compound
                            ):
                                subj_compound.append(tkk)
                        for tkk in tk.children:
                            if (
                                tkk.dep_
                                in [
                                    "compound",
                                    "poss",
                                    "amod",
                                    "det",
                                    "aux",
                                ]
                                and tkk not in subj_compound
                                and tkk not in compound
                            ):
                                subj_compound.append(tkk)
                        if tk not in subj_compound:
                            subj_compound.append(tk)
                        subj_compound.extend(compound)
                        # remove tokens with duplicate indexes from token.i
                        for sub in subj_compound:
                            if subj_compound.count(sub) > 1:
                                subj_compound.remove(sub)

                        subj_compound = sorted(subj_compound, key=lambda x: x.i)
                        print(f"subj_compound: {subj_compound}")
                        # create a string that joins each word in token.sent with a space but only if the word is in subj_compound in order of each word's index
                        subj_text = " ".join(
                            [
                                token.text
                                for token in token.sent
                                if token in subj_compound
                            ]
                        )
                        print(f"Subject: {subj_text}")
                        if tk.dep_ == "prep":
                            for tkk in tk.children:
                                if tkk.dep_ == "pobj":
                                    if tkk not in subj_compound:
                                        subj_compound.append(tkk)
                                    for tkkk in tkk.children:
                                        if tkkk not in subj_compound:
                                            subj_compound.append(tkkk)
                    else:
                        if tk.dep_ in [
                            "compound",
                            "poss",
                            "amod",
                            "det",
                            "aux",
                        ]:
                            subj_compound.append(tk)
                        subj_compound.append(tk)
                        subj_compound = sorted(subj_compound, key=lambda x: x.i)
                        subj_text = " ".join(
                            [
                                token.text
                                for token in token.sent
                                if token in subj_compound
                            ]
                        )
                        print(f"Subject: {subj_text}")
                        if tk.dep_ == "prep":
                            for tkk in tk.children:
                                if tkk.dep_ == "pobj":
                                    if tkk not in subj_compound:
                                        subj_compound.append(tkk)
                                    for tkkk in tkk.children:
                                        if tkkk not in subj_compound:
                                            subj_compound.append(tkkk)
                subj_compound = sorted(subj_compound, key=lambda x: x.i)
                print(f"subj_compound: {subj_compound}")
                for tk in subj_compound:
                    if tk.text not in subj_text.split():
                        print(f"Fixing missing subj compound token at token: {tk.text}")
                        temp = subj_compound
                        temp.append(subj[0])
                        temp = sorted(temp, key=lambda x: x.i)
                        subj_text = " ".join(
                            [token.text for token in token.sent if token in temp]
                        )
                if subj[0].text not in subj_text.split():
                    print("Adding subject to subj_compound and subj_text")
                    if subj[0] not in subj_compound:
                        subj_compound.append(subj[0])
                    subj_text = " ".join(
                        [token.text for token in token.sent if token in subj_compound]
                    )

                if subj_text == "" or subj_text == " ":
                    subj_text = subj[0].text
                    print(f"Subject: {subj_text}")
                print(f"Subject: {subj_text}")
                print(f"Tokens children: {[tk.text for tk in token.children]}")

                dobj = [
                    tk
                    for tk in token.children
                    if tk.dep_ == "dobj" or tk.dep_ == "pobj"
                ]
                if (
                    len(dobj) == 0
                    and token.dep_ == "conj"
                    and token.tag_ in ["VERB", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]
                ):
                    for tk in token.children:
                        if tk.dep_ in ["dobj", "pobj"]:
                            dobj = [tk]
                    else:
                        dobj = [token]
                if len(dobj) == 0:
                    dobj = [tk for tk in token.children if tk.dep_ == "prep"]
                    for tk in dobj:
                        for tkk in tk.children:
                            if tkk.dep_ == "pobj":
                                dobj = [tkk]
                                for tkkk in tkk.children:
                                    if tkkk not in dobj:
                                        dobj.append(tkkk)
                if len(dobj) == 0:

                    dobj = [
                        tk
                        for tk in token.head.children
                        if tk.dep_ == "attr"
                        and (subj[0].head == tk.head or token == tk.head)
                    ]
                    print(f"Last resort dobj: {dobj}")
                else:
                    dobj = sorted(dobj, key=lambda x: x.i)
                    obj_compound.extend(ob for ob in dobj)
                # if len(dobj) == 0:
                dobj = sorted(dobj, key=lambda x: x.i)
                for tok in token.children:

                    if tok.dep_ == "prep":
                        dobj = [tok]
                        for tkk in tok.children:
                            if tkk.dep_ == "pobj":
                                if tkk not in dobj:
                                    dobj.append(tkk)
                                for tkkk in tkk.children:
                                    if tkkk not in dobj and tkkk.dep_ in [
                                        "compound",
                                        "amod",
                                        "poss",
                                        "det",
                                    ]:
                                        dobj.append(tkkk)

                    elif tok.dep_ == "xcomp":

                        # Find main clause via the ROOT verb of the whole sentence
                        root_verb = None
                        for tk in docs:
                            if tk.dep_ == "ROOT":
                                root_verb = tk
                                break

                        assert root_verb is not None
                        ob = [
                            tkk
                            for tkk in tok.children
                            if tkk.dep_ == "pobj" or tkk.dep_ == "dobj"
                        ]

                        if len(ob) > 0 and len(dobj) == 0:
                            dobj = ob
                        elif len(dobj) > 0 and len(ob) > 0:
                            dobj.extend(ob)
                        elif len(dobj) == 0 and len(ob) == 0:
                            if tok.head == root_verb or tok.head.tag_ == "VBG":
                                if len(dobj) == 0:
                                    dobj = [
                                        tk for tk in tok.children if tk.dep_ == "dobj"
                                    ]

                dobj = sorted(dobj, key=lambda x: x.i)
                for objj in dobj:
                    if objj.dep_ not in ["conj", "cc"] and objj not in obj_compound:
                        obj_compound.append(objj)
                    for objjj in objj.children:
                        if objjj not in obj_compound and objjj.dep_ in [
                            "compound",
                            "amod",
                            "poss",
                            "det",
                        ]:
                            obj_compound.append(objjj)
                        if objjj.dep_ == "prep":
                            for objjjj in objjj.children:
                                if objjjj.dep_ in [
                                    "pobj",
                                    "det",
                                    "amod",
                                    "poss",
                                    "compound",
                                ]:
                                    if objjjj not in obj_compound:
                                        obj_compound.append(objjjj)
                                    for objjjjj in objjjj.children:
                                        if objjjjj not in obj_compound:
                                            obj_compound.append(objjjjj)

                    if objj.dep_ == "conj":
                        for objjj in objj.head.children:
                            if (
                                objjj.dep_
                                in [
                                    "compound",
                                    "amod",
                                    "poss",
                                    "det",
                                ]
                                and objjj not in obj_compound
                                and objjj not in subj_compound
                            ):

                                obj_compound.append(objjj)

                break
        dobj = sorted(dobj, key=lambda x: x.i)
        obj_compound = sorted(obj_compound, key=lambda x: x.i)
        if len(dobj) < 1:
            print(f"DOBJ still seems to be empty. Token: {token}")
            if docs[token.i - 1].text == "to":  # or token.tag_ == "VBG":
                if token.head.pos_ == "VERB":
                    dobj = [token.head]
            else:
                for tkn in token.children:
                    if tkn.dep_ == "xcomp" or tkn.dep_ == "advcl":
                        if docs[tkn.i - 1].text == "to" or tkn.tag_ == "VBG":
                            if tkn.head.pos_ == "VERB":
                                dobj = [tkn.head]
        dobj = sorted(dobj, key=lambda x: x.i)
        if len(dobj) < 1:

            if token.dep_ == "conj":
                print(f"DOBJ still seems to be empty. Token: {token}")
                for tk in token.children:
                    print(f"Token: {tk.text}, Dependency: {tk.dep_}")
                    if tk.dep_ in ["dobj", "pobj"]:
                        dobj = [tk]
                if len(dobj) == 0:
                    dobj = [token]
            elif len(dobj) == 0:
                print(
                    f"dobj seems to still be empty so we will make it the token: {token}"
                )
                dobj = [token]
            # elif len(dobj) == 0:
            #     dobj = [token.head]
        for tk in dobj:
            if (
                tk.text == action.text
                and action.dep_ == "ROOT"
                and "mark" in {child.dep_ for child in token.children}
            ):
                dobj.remove(tk)
                dobj.append(nlp("something")[0])
        dobj = sorted(dobj, key=lambda x: x.i)
        print(f"Subject: {subj}, Action: {action}, Object: {dobj}")
        if len(dobj) > 1:
            # Since dobj is more than 1, we must determine which object in dobj is correctly associated with the action by checking the dependency relation in a more complex way than simply checking the dep tag
            # We can do this by checking the dependency relation of each token in dobj to the action token and selecting the token with the closest dependency relation to the action token
            # We can also check the dependency relation of each token in dobj to the ROOT verb to ensure that the object is correctly associated with the action
            # We can also check the dependency relation of each token in dobj to the subject token to ensure that the object is correctly associated with the action
            # We can also check the dependency relation of each token in dobj to the auxiliaries to ensure that the object is correctly associated with the action
            # We can also check the dependency relation of each token in dobj to the verb's children to ensure that the object is correctly associated with the action
            # We can also check the dependency relation of each token in dobj to the verb's head to ensure that the object is correctly associated with the action
            # The actions are typically the roots of the sentence, but they can also be other verbs
            action_tokens = [
                tok for tok in docs if tok.dep_ in {"ROOT", "relcl", "xcomp", "ccomp"}
            ]

            # The verbs are typically the tokens with part-of-speech "VERB", but they can also be other tokens with verb-like behavior
            verb_tokens = [tok for tok in docs if tok.pos_ in {"VERB", "AUX"}]

            # The subjects are typically the tokens with a direct dependency to a verb, but they can also be other tokens
            subject_tokens = [
                tok
                for tok in docs
                if any(
                    sub.dep_ in {"nsubj", "nsubjpass", "csubj", "csubjpass"}
                    for sub in tok.children
                )
            ]

            # The auxiliaries are typically the tokens with dependency "aux", but they can also be other tokens
            auxiliaries_tokens = [tok for tok in docs if tok.dep_ in {"aux", "auxpass"}]

            # The children of the verbs, including indirect children
            verbs_children_tokens = [
                child for verb in verb_tokens for child in verb.subtree
            ]

            # The heads of the verbs, including indirect heads
            verbs_head_tokens = [
                ancestor for verb in verb_tokens for ancestor in verb.ancestors
            ]

            for tk in dobj:
                print(
                    f"Token: {tk.text}, Dependency to Action: {[tkk.dep_ for tkk in action_tokens if tkk.head == tk]}"
                )
                print(
                    f"Token: {tk.text}, Dependency to ROOT Verb: {[tkk.dep_ for tkk in verb_tokens if tkk.head == tk and tkk.dep_ == 'ROOT']}"
                )
                print(
                    f"Token: {tk.text}, Dependency to Subject: {[tkk.dep_ for tkk in subject_tokens if tkk.head == tk]}"
                )
                print(
                    f"Token: {tk.text}, Dependency to Auxiliaries: {', '.join([aux.dep_ for aux in auxiliaries_tokens if tk.head == aux]) or 'N/A'}"
                )
                print(
                    f"Token: {tk.text}, Dependency to Verb's Children: {', '.join([child.dep_ for child in verbs_children_tokens if tk.head == child]) or 'N/A'}"
                )
                print(
                    f"Token: {tk.text}, Dependency to Verb's Head: {', '.join([ancestor.dep_ for ancestor in verbs_head_tokens if tk.head == ancestor]) or 'N/A'}"
                )
                if tk not in obj_compound:
                    obj_compound.append(tk)
                for tko in tk.children:
                    if (
                        tko.dep_ == "compound"
                        or tko.dep_ == "poss"
                        or tko.dep_ == "amod"
                        or tko.dep_ == "det"
                        or tko.dep_ == "attr"
                    ):
                        if len(obj_compound) == 0:
                            obj_compound = [tk]
                        else:
                            if tko not in obj_compound:
                                obj_compound.append(tk)
                        if tko.dep_ == "attr":
                            for tkk in tko.children:
                                if tkk.i not in [tkk.i for tkk in obj_compound]:
                                    obj_compound.append(tkk)
                    elif tko.dep_ == "prep":
                        obj_compound = [tk]
                        for tkk in tko.children:
                            if tkk.dep_ == "pobj":
                                if tkk not in obj_compound:
                                    obj_compound.append(tkk)
                                for tkkk in tkk.children:
                                    if tkkk not in obj_compound:
                                        obj_compound.append(tkkk)

        for tk in dobj[0].children:
            print(f"dobjs children: Token: {tk.text}, Dependency: {tk.dep_}")
            if (
                tk.dep_ == "compound"
                or tk.dep_ == "poss"
                or tk.dep_ == "amod"
                or tk.dep_ == "det"
                or tk.dep_ == "attr"
            ):
                if tk not in obj_compound:
                    obj_compound.append(tk)
                    # print(f"added to obj_compound: {tk.text}")
                if tk.dep_ == "attr":
                    for tkk in tk.children:
                        if tkk not in obj_compound:
                            obj_compound.append(tkk)
                        for tkkk in tkk.children:
                            if tkkk not in obj_compound:
                                obj_compound.append(tkkk)
            elif tk.dep_ == "prep":
                if len(obj_compound) == 0:
                    obj_compound = [tk]
                else:
                    obj_compound.append(tk)
                for tkk in tk.children:
                    if tkk.dep_ == "pobj":
                        if tkk not in obj_compound:
                            obj_compound.append(tkk)
                        for tkkk in tkk.children:
                            if tkkk not in obj_compound:
                                obj_compound.append(tkkk)
                    elif tkk.dep_ == "prep":
                        if len(obj_compound) == 0:
                            obj_compound = [tk]
                        else:
                            obj_compound.append(tk)
                        for tkkk in tkk.children:
                            if tkkk.dep_ == "pobj":
                                if tkkk not in obj_compound:
                                    obj_compound.append(tkkk)
                                for tkkkk in tkkk.children:
                                    if tkkkk not in obj_compound:
                                        obj_compound.append(tkkk)
        obj_compound = sorted(obj_compound, key=lambda x: x.i)
        print(f"obj_compound: {obj_compound}")
        if len(obj_compound) > 0 and "prep" not in {tk.dep_ for tk in action.children}:
            if dobj[0] not in obj_compound:
                obj_compound.append(dobj[0])
            # Ensure no token is repeated in obj_compound that was in subj_compound (unless the token exists twice in the sentence or had a different index)
            obj_compound = [
                tk
                for tk in obj_compound
                if tk not in subj_compound
                or (
                    tk in subj_compound
                    and sum(tk.text == t.text for t in subj_compound) > 1
                )
                or tk.i != [t.i for t in subj_compound if t.text == tk.text][0]
            ]

        elif len(obj_compound) == 0:
            print(f"For some reason, obj_compound is empty. Token: {token}")
            obj_compound = dobj
        elif len(obj_compound) > 0 and "prep" in {tk.dep_ for tk in action.children}:
            for tk in action.children:
                if tk.dep_ in ["det", "amod", "poss", "compound", "attr"]:
                    if tk not in obj_compound:
                        obj_compound.append(tk)
                if tk.dep_ == "prep":
                    if len(obj_compound) == 0:
                        obj_compound = [tk]
                    else:
                        if tk not in obj_compound:
                            obj_compound.append(tk)
                    for tkk in tk.children:
                        if tkk.dep_ == "pobj":
                            if tkk not in obj_compound:
                                obj_compound.append(tkk)
                            for tkkk in tkk.children:
                                if tkkk not in obj_compound:
                                    obj_compound.append(tkkk)
                        elif tkk.dep_ == "prep":
                            if len(obj_compound) == 0:
                                obj_compound = [tk]
                            else:
                                obj_compound.append(tk)
                            for tkkk in tkk.children:
                                if tkkk.dep_ == "pobj":
                                    if tkkk not in obj_compound:
                                        obj_compound.append(tkkk)
                                    for tkkkk in tkkk.children:
                                        if tkkkk not in obj_compound:
                                            obj_compound.append(tkkk)
        obj_compound = sorted(obj_compound, key=lambda x: x.i)

        action_compound = [action]
        action_compound.extend(
            action_tok
            for action_tok in token.children
            if action_tok.dep_
            in [
                "aux",
                "auxpass",
                "neg",
                "prt",
                "det",
                "amod",
                "poss",
                "compound",
                "pos",
                # "mark",
                "xcomp",
            ]
        )
        if token.head.dep_ == "ROOT":
            action_compound.extend(
                action_tok
                for action_tok in token.head.children
                if action_tok.dep_
                in [
                    "aux",
                    "auxpass",
                    "neg",
                    "prt",
                    "det",
                    "amod",
                    "poss",
                    "compound",
                    "pos",
                    # "mark",
                    "xcomp",
                ]
                if action_tok.i > token.head.i
            )
        if (
            token.tag_
            in [
                "VERB",
                "VB",
                "VBD",
                "VBG",
                "VBN",
                "VBP",
                "VBZ",
            ]
            and token.head.dep_ != "xcomp"
        ):
            if token.text == dobj[0].text:
                if dobj[0] in obj_compound:
                    dobj_text = ""
                else:
                    print(f"obj_compound: {obj_compound}")
                    print(f"Token: {token.text}, dobj: {dobj[0].text}")
        if (
            token.dep_ == "conj"
            and token.head.dep_ == "xcomp"
            and token.tag_
            in [
                "VERB",
                "VB",
                "VBD",
                "VBG",
                "VBN",
                "VBP",
                "VBZ",
            ]
        ):
            action_compound.extend(
                action_tok
                for action_tok in token.head.children
                if action_tok.dep_
                in [
                    "aux",
                    "auxpass",
                    "neg",
                    "prt",
                    "det",
                    "amod",
                    "poss",
                    "compound",
                    "pos",
                    # "mark",
                    "xcomp",
                ]
            )
            action_compound.remove(action)
            action_compound.append(root_verb)
            if root_verb in obj_compound:
                obj_compound.remove(root_verb)
                print(f"obj_compound: {obj_compound}")
        for action_tok in action_compound:
            if action_tok in obj_compound:
                for obj in obj_compound:
                    if action_tok.i == obj.i:
                        obj_compound.remove(obj)
                        print(f"obj_compound: {obj_compound}")
        obj_compound = sorted(obj_compound, key=lambda x: x.i)
        for tk in obj_compound:
            print(f"obj_compound: {tk.text}, index: {tk.i}")
        # create a string that joins each word in token.sent with a space but only if the word is in subj_compound in order of each word's index
        dobj_text = " ".join([token.text for token in obj_compound])
        print(f"obj_compound: {obj_compound}")
        print(f"dobj_text: {dobj_text}")
        # Ensure no token is repeated in action_compound that was in subj_compound or obj_compound (unless the token exists twice in the sentence or had a different index)
        # if "mark" in {tk.dep_ for tk in action.children}:
        #         # Make the action compund consist of only mark token, its associated verb and the verb's children
        #         action_compound = [
        #             tk
        #             for tk in action.children
        #         ]
        #         action_compound.append(action)

        if "xcomp" in {tk.dep_ for tk in action_compound}:
            for tk in action_compound:
                if tk.dep_ == "xcomp":
                    action_compound.extend(
                        tkk
                        for tkk in tk.children
                        if tkk.dep_
                        in [
                            "aux",
                            "auxpass",
                            "neg",
                            "prt",
                            "det",
                            "amod",
                            "poss",
                        ]
                    )
                    break

        if docs[subj[0].i + 1].dep_ == "aux":
            action_compound.append(docs[subj[0].i + 1])

        action_compound = [
            tk
            for tk in action_compound
            if tk not in subj_compound
            or (
                tk in subj_compound
                and sum(tk.text == t.text for t in subj_compound) > 1
            )
            or tk.i != [t.i for t in subj_compound if t.text == tk.text][0]
        ]

        # action_compound = [
        #     tk
        #     for tk in action_compound
        #     if tk not in obj_compound
        #     or (
        #         tk in obj_compound
        #         and sum(tk.text == t.text for t in obj_compound) > 1
        #     )
        #     or tk.i != [t.i for t in obj_compound if t.text == tk.text][0]
        # ]
        for tk in subj_compound:
            if tk.dep_ == "acomp" and token.tag_ == "VBP":
                action_compound.append(tk)
                if tk in subj_compound:
                    subj_compound.remove(tk)
                subj_text = subj[0].text if subj else ""
                print(f"subj_compound: {subj_compound}")
                print(f"obj_compound: {obj_compound}")
                print(f"action_compound: {action_compound}")
                print(f"Subject: {subj_text}")
                subj_compound = sorted(subj_compound, key=lambda x: x.i)
                subj_text = " ".join(
                    [token.text for token in token.sent if token in subj_compound]
                )
                print(f"Subject: {subj_text}")
        action_compound = sorted(action_compound, key=lambda x: x.i)
        action_text = " ".join(
            [token.text for token in token.sent if token in action_compound]
        )
        print(f"Action Compound: {action_compound}")
        template = ""
        if subj[0].i < dobj[0].i:
            template = f"{subj_text} {action_text} {dobj_text}"
        elif subj[0].i > dobj[0].i:
            template = f"{dobj_text} {subj_text} {action_text}"
        print(f"\n \n Template: {template}\n \n")
        templates.append(template)
        for tk in docs:
            if (
                tk.dep_ == "conj"
                and tk.pos_ in ["NOUN", "PROPN"]
                and (tk.head.text == action.text or tk.head.dep_ == "ROOT")
                and tk not in subj_compound
                and tk not in obj_compound
            ):
                dobj = [tk]
                obj_compound = dobj

                for tkk in tk.children:
                    if tkk.dep_ in ["compound", "amod", "poss", "det", "attr"]:
                        obj_compound.append(tkk)
                    if tkk.dep_ == "prep":
                        for tkkk in tkk.children:
                            if tkkk.dep_ == "pobj":
                                if tkkk not in obj_compound:
                                    obj_compound.append(tkkk)
                                for tkkkk in tkkk.children:
                                    if tkkkk not in obj_compound:
                                        obj_compound.append(tkkk)
                            if tkkk.dep_ == "prep":
                                for tkkk in tkk.children:
                                    if tkkk.dep_ == "pobj":
                                        if tkkk not in obj_compound:
                                            obj_compound.append(tkkk)
                                        for tkkkk in tkkk.children:
                                            if tkkkk not in obj_compound:
                                                obj_compound.append(tkkk)

                dobj_text = " ".join(
                    [token.text for token in token.sent if token in obj_compound]
                )
                print(f"Subject: {subj}, Action: {action}, Object: {dobj}")
                print(f"Action Compound: {action_compound}")
                template = f"{subj_text} {action_text} {dobj_text}"
                print(f"\n \n Template (C1): {template}\n \n")
                templates.append(template)
            elif (
                tk.dep_ == "conj"
                and tk.pos_ in ["NOUN", "PROPN"]
                and dobj[0].head.text == action.text
                and dobj[0] != action
                and (tk.head.head.dep_ == "ROOT" or tk.head.head.text == action.text)
                and tk not in subj_compound
                and tk not in obj_compound
                and docs[tk.i - 2].dep_ != "nsubj"
            ):
                dobj = [tk]
                obj_compound = dobj
                for tkk in tk.children:
                    if tkk.dep_ in ["compound", "amod", "poss", "det", "attr"]:
                        obj_compound.append(tkk)
                    if tkk.dep_ == "prep":
                        for tkkk in tkk.children:
                            if tkkk.dep_ == "pobj":
                                if tkkk not in obj_compound:
                                    obj_compound.append(tkkk)
                                for tkkkk in tkkk.children:
                                    if tkkkk not in obj_compound:
                                        obj_compound.append(tkkk)
                            if tkkk.dep_ == "prep":
                                for tkkk in tkk.children:
                                    if tkkk.dep_ == "pobj":
                                        if tkkk not in obj_compound:
                                            obj_compound.append(tkkk)
                                        for tkkkk in tkkk.children:
                                            if tkkkk not in obj_compound:
                                                obj_compound.append(tkkk)
                dobj_text = " ".join(
                    [token.text for token in token.sent if token in obj_compound]
                )
                print(f"dobj: {dobj}")

                print(f"Subject: {subj}, Action: {action}, Object: {dobj}")
                print(f"Action Compound: {action_compound}")
                print(f"Object Compound: {obj_compound}")
                print(f"dobj_text: {dobj_text}")
                template = f"{subj_text} {action_text} {dobj_text}"
                print(f"\n \n Template (C2): {template}\n \n")
                templates.append(template)
            elif (
                tk.dep_ == "conj"
                and tk.pos_ in ["NOUN", "PROPN"]
                and (tk.head.dep_ == "xcomp" or tk.head.head.dep_ == "xcomp")
            ):

                dobj = [tk]
                obj_compound = dobj
                for tkk in tk.children:
                    if tkk.dep_ in ["compound", "amod", "poss", "det", "attr"]:
                        obj_compound.append(tkk)
                    if tkk.dep_ == "prep":
                        for tkkk in tkk.children:
                            if tkkk.dep_ == "pobj":
                                if tkkk not in obj_compound:
                                    obj_compound.append(tkkk)
                                for tkkkk in tkkk.children:
                                    if tkkkk not in obj_compound:
                                        obj_compound.append(tkkk)
                            if tkkk.dep_ == "prep":
                                for tkkk in tkk.children:
                                    if tkkk.dep_ == "pobj":
                                        if tkkk not in obj_compound:
                                            obj_compound.append(tkkk)
                                        for tkkkk in tkkk.children:
                                            if tkkkk not in obj_compound:
                                                obj_compound.append(tkkk)
                dobj_text = " ".join(
                    [token.text for token in token.sent if token in obj_compound]
                )
                # print(f"Subject: {subj}, Action: {action}, Object: {dobj}")
                # print(f"Action Compound: {action_compound}")
                template = f"{subj_text} {action_text} {dobj_text}"
                print(f"\n \n Template (C3): {template}\n \n")
                templates.append(template)
            elif (
                tk.dep_ == "conj"
                and tk.pos_ in ["NOUN", "PROPN"]
                and tk.head.dep_ == "conj"
                and tk.head.text == action.text
                and dobj[0] != action
            ):

                dobj = [tk]
                obj_compound = dobj
                for tkk in tk.children:
                    if tkk.dep_ in ["compound", "amod", "poss", "det", "attr"]:
                        obj_compound.append(tkk)
                    if tkk.dep_ == "prep":
                        for tkkk in tkk.children:
                            if tkkk.dep_ == "pobj":
                                if tkkk not in obj_compound:
                                    obj_compound.append(tkkk)
                                for tkkkk in tkkk.children:
                                    if tkkkk not in obj_compound:
                                        obj_compound.append(tkkk)
                            if tkkk.dep_ == "prep":
                                for tkkk in tkk.children:
                                    if tkkk.dep_ == "pobj":
                                        if tkkk not in obj_compound:
                                            obj_compound.append(tkkk)
                                        for tkkkk in tkkk.children:
                                            if tkkkk not in obj_compound:
                                                obj_compound.append(tkkk)

                dobj_text = " ".join(
                    [token.text for token in token.sent if token in obj_compound]
                )
                # print(f"Subject: {subj}, Action: {action}, Object: {dobj}")
                # print(f"Action Compound: {action_compound}")
                template = f"{subj_text} {action_text} {dobj_text}"
                print(f"\n \n Template (C4): {template}\n \n")
                templates.append(template)
            elif (
                tk.dep_ == "conj"
                and tk.pos_ in ["NOUN", "PROPN"]
                and tk.head.text == action.text
                and dobj[0] != action
            ):

                dobj = [tk]
                obj_compound = dobj
                for tkk in tk.children:
                    if tkk.dep_ in ["compound", "amod", "poss", "det", "attr"]:
                        obj_compound.append(tkk)
                    if tkk.dep_ == "prep":
                        for tkkk in tkk.children:
                            if tkkk.dep_ == "pobj":
                                if tkkk not in obj_compound:
                                    obj_compound.append(tkkk)
                                for tkkkk in tkkk.children:
                                    if tkkkk not in obj_compound:
                                        obj_compound.append(tkkk)
                            if tkkk.dep_ == "prep":
                                for tkkk in tkk.children:
                                    if tkkk.dep_ == "pobj":
                                        if tkkk not in obj_compound:
                                            obj_compound.append(tkkk)
                                        for tkkkk in tkkk.children:
                                            if tkkkk not in obj_compound:
                                                obj_compound.append(tkkk)

                dobj_text = " ".join(
                    [token.text for token in token.sent if token in obj_compound]
                )
                # print(f"Subject: {subj}, Action: {action}, Object: {dobj}")
                # print(f"Action Compound: {action_compound}")
                # print(f"Object Compound: {obj_compound}")
                template = f"{subj_text} {action_text} {dobj_text}"
                print(f"\n \n Template (C5): {template}\n \n")
                templates.append(template)
            elif (
                tk.dep_ == "conj"
                and tk.pos_ in ["NOUN", "PROPN"]
                and tk.head.dep_ == "conj"
                and (
                    tk.head.head.text == action.text
                    or tk.head.head.head.text == action.text
                )
                and dobj[0] != action
            ):

                dobj = [tk]
                obj_compound = dobj
                for tkk in tk.children:
                    if tkk.dep_ in ["compound", "amod", "poss", "det", "attr"]:
                        obj_compound.append(tkk)
                    if tkk.dep_ == "prep":
                        for tkkk in tkk.children:
                            if tkkk.dep_ == "pobj":
                                if tkkk not in obj_compound:
                                    obj_compound.append(tkkk)
                                for tkkkk in tkkk.children:
                                    if tkkkk not in obj_compound:
                                        obj_compound.append(tkkk)
                            if tkkk.dep_ == "prep":
                                for tkkk in tkk.children:
                                    if tkkk.dep_ == "pobj":
                                        if tkkk not in obj_compound:
                                            obj_compound.append(tkkk)
                                        for tkkkk in tkkk.children:
                                            if tkkkk not in obj_compound:
                                                obj_compound.append(tkkk)

                dobj_text = " ".join(
                    [token.text for token in token.sent if token in obj_compound]
                )
                # print(f"Subject: {subj}, Action: {action}, Object: {dobj}")
                # print(f"Action Compound: {action_compound}")
                template = f"{subj_text} {action_text} {dobj_text}"
                print(f"\n \n Template (C6): {template}\n \n")
                templates.append(template)

            elif (
                tk.dep_ == "conj"
                and tk.pos_ in ["NOUN", "PROPN"]
                and tk.head.dep_ == "conj"
                and action.text in [tk.text for tk in tk.ancestors]
                and dobj[0] != action
            ):
                other_verb = False
                for right_token in docs[token.i + 1 :]:
                    if right_token.dep_ == "conj":

                        if right_token.pos_ == "VERB":
                            print(f"Token: {tk.text}, Right: {tk.rights}")
                            other_verb = True
                            break
                if not other_verb:
                    dobj = [tk]
                    obj_compound = dobj
                    for tkk in tk.children:
                        if tkk.dep_ in [
                            "compound",
                            "amod",
                            "poss",
                            "det",
                            "attr",
                        ]:
                            obj_compound.append(tkk)
                        if tkk.dep_ == "prep":
                            for tkkk in tkk.children:
                                if tkkk.dep_ == "pobj":
                                    if tkkk not in obj_compound:
                                        obj_compound.append(tkkk)
                                    for tkkkk in tkkk.children:
                                        if tkkkk not in obj_compound:
                                            obj_compound.append(tkkk)
                                if tkkk.dep_ == "prep":
                                    for tkkk in tkk.children:
                                        if tkkk.dep_ == "pobj":
                                            if tkkk not in obj_compound:
                                                obj_compound.append(tkkk)
                                            for tkkkk in tkkk.children:
                                                if tkkkk not in obj_compound:
                                                    obj_compound.append(tkkk)

                    dobj_text = " ".join(
                        [token.text for token in token.sent if token in obj_compound]
                    )
                    # print(f"Subject: {subj}, Action: {action}, Object: {dobj}")
                    # print(f"Action Compound: {action_compound}")
                    template = f"{subj_text} {action_text} {dobj_text}"
                    print(f"\n \n Template (C7): {template}\n \n")
                    templates.append(template)

        assert len(subj) > 0 and len(action_compound) > 0
    return templates


if __name__ == "__main__":

    # Example usage
    queries = [
        "Where was the World Cup held in 2018?",
        "When will the 2022 Winter Olympics be held?",
        "I think someone is planning a surprise party",
        "I need to think of a popular tourist attraction",
        "I am planning a trip to Europe",
        "Who is learning to play the guitar?",
        "I think someone is studying for a Chemistry test",
        "What product should I sell in my electronics store?",
        "What should I eat at the French restaurant?",
        "What book should I read?",
        "Where should I go for a night out to have drinks and meet someone?",
        "What bar should I avoid?",
        "What fashion accessory should I wear to the party?",
        "What is the future of transportation?",
        "What is the current state of the ebola outbreak?",
        "I am a farmer, what crop should I grow to make the most profit?",
        "Is the new iPhone worth buying?",
        "As a farmer, what crop would be make the most money?",
        "What farm vegetable is selling the most these days?",
    ]
    for query in queries:
        print(f"Query: {query}")
        templates = generate_templates(query)
        print(f"Templates: {templates}")
        print("\n")
