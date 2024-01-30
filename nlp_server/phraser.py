#!/usr/bin/env python
import logging
from typing import List
from timeit import default_timer

from flask import make_response
from flask_jsonpify import jsonify
from flask_restful import reqparse
from flask_restful import Resource
from spacy.tokens import Doc
from werkzeug.datastructures import FileStorage

from models.boolq_utils import Question2Sentence
from collections import defaultdict

# from .modules import yake
# from .modules.rake import Metric
# from .modules.rake import Rake

# import neuralcoref

# Python 3
unicode_ = str

PERMITTED_OPERATIONS = [
    "get_main_subject",
    "get_resolved_text",
    "extract_phrases",
    "get_filtered_phrases",
    "extract_keywords",
    "get_pure_rake",
    "get_pure_yake",
    "get_combination",
    "get_doc_ents",
    "convert_question_to_sentence",
    "get_doc_tags",
]


class Phraser(Resource):
    def __init__(
        self,
        nlp,
        bio_nlp, 
        batch_size=50,
        n_process=1,  # number of processors to use
        # neuralcoref params
        greedyness=0.5,
        max_dist=50,
        max_dist_match=500,
        blacklist=None,
        store_scores=True,
        conv_dict=None,
    ):

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        logging.getLogger("spacy").setLevel(logging.ERROR)

        self.nlp = nlp
        self.bio_nlp = bio_nlp
        self.greedyness = greedyness
        self.max_dist = max_dist
        self.max_dist_match = max_dist_match
        self.blacklist = blacklist
        self.store_scores = store_scores
        self.conv_dict = conv_dict
        self.stopwords = ["today", "already", "having", "also", "comprised"]

        self.req_parser = reqparse.RequestParser()
        self.req_parser.add_argument(
            "option",
            type=str,
            help="The method to call (get_main_subject, extract_phrases, extract_keywords, get_resolved_text, get_filtered_phrases)",
        )
        self.req_parser.add_argument(
            "text",
            type=FileStorage,
            location="json",
            help="The text to process",
        )  # should only use type = list when location = json
        self.req_parser.add_argument(
            "batch_size",
            type=int,
            help="Batch size to use for NLP processing",
        )
        self.req_parser.add_argument(
            "n_process",
            type=int,
            help="Number of processors to use for NLP processing",
        )
        self.req_parser.add_argument(
            "keywords",
            type=FileStorage,
            location="json",
            help="A list of keyword(s) to search",
        )
        self.req_parser.add_argument(
            "texts",
            type=str,
            action="append",
            help="A list of keyword(s) to search",
        )
        self.req_parser.add_argument(
            "domain",
            type=str,
            help="Domain specific NLP engine to load",
        )
        # if 'neuralcoref' not in self.nlp.pipe_names:
        #     neuralcoref.add_to_pipe(self.nlp,
        #         greedyness=self.greedyness,
        #         max_dist=self.max_dist,
        #         max_dist_max=self.max_dist_match,
        #         blacklist=self.blacklist,
        #         store_scores=self.store_scores,
        #         conv_dict=self.conv_dict
        #         )
        if "merge_entities" not in self.nlp.pipe_names:
            merge_ents = self.nlp.create_pipe("merge_entities")
            # self.nlp.add_pipe(merge_ents, after="ner")
        try:
            Doc.set_extension("extract_phrases", method=self.extract_phrases)
            Doc.set_extension("get_main_subject", method=self.get_main_subject)
            Doc.set_extension("get_resolved_text", method=self.get_resolved_text)
            Doc.set_extension("get_filtered_phrases", method=self.get_filtered_phrases)
            Doc.set_extension("extract_keywords", method=self.extract_keywords)
            Doc.set_extension("get_pure_rake", method=self.get_pure_rake)
            Doc.set_extension("get_pure_yake", method=self.get_pure_yake)
            Doc.set_extension("get_combination", method=self.get_combination)
            Doc.set_extension("get_doc_ents", method=self.get_doc_ents)
            Doc.set_extension("get_doc_tags", method=self.get_doc_tags)
            Doc.set_extension(
                "convert_question_to_sentence", method=self.convert_question_to_sentence
            )
        except ValueError:
            pass

    def convert_question_to_sentence(self, texts):
        question_to_sentence = Question2Sentence(self.nlp)
        return [question_to_sentence.run_local(x) for x in texts]

    def get_doc_tags(self, texts):
        docs = self.nlp.pipe(texts, disable=["parser"])
        result = []
        for doc in docs:
            ents = [(ent.text, ent.label_) for ent in doc.ents]
            pos_tags = [{'text': token.text,
                         'lemma': token.lemma_,
                         'pos': token.pos_,
                         'pos_tag': token.tag_,
                         'dep': token.dep_,
                         'is_alpha': token.is_alpha,
                         'is_stop': token.is_stop
                         } for token in doc]
            result.append({"ner": ents, "tags": pos_tags})
        return result

    def get_doc_ents(self, texts, domain="general"):

        def inside(a, b):   # to check whether interval a :( x1, y1) is inside b :(x2, y2)
            if a == b:
                return False
            elif (a[0] >=b[0]) and (a[1] <= b[1]): # a is inside b
                return True
            
        def prune(entities, span_entities):
            keys = list( span_entities.keys() )
            for i, span in enumerate(keys):
                for j, span_ in enumerate(keys):
                    if i != j:
                        if inside(span, span_):
                            try:
                                del entities[span_entities[span]]
                            except:
                                continue
            return entities
                
        if domain == "biology":
            docs = [
                [
                    nlp(text) for nlp in self.bio_nlp
                ] for text in texts
            ]
            
            output = []
            # priority is with models that come later
            for documents in docs:
                entities = defaultdict(lambda: [])
                span_entity = {}
                for doc in documents:
                    for ent in doc.ents:
                        entities[ent.text] = list({self.convert_medical_label(ent.label_)}.union(entities[ent.text]))
                        
                        span_entity[(ent.start_char, ent.end_char)] = ent.text

                output.append(list(prune(entities, span_entity).items()))
            return output
        
        else:

            docs = self.nlp.pipe(texts, disable=["tagger", "parser"])
            return [
                [(ent.text, self.convert_label(ent.label_)) for ent in doc.ents]
                for doc in docs
            ]


    def convert_label(self, ent_label):
        # non confirming Spacy -> Roth will be OTHERS
        # skip LAW, it only maps to 'ENTY:other'

        new_ent_label = ["OTHERS"]

        if "GPE" == ent_label:
            new_ent_label = ["LOC:state", "LOC:city", "LOC:country"]

        elif "LOC" == ent_label:
            new_ent_label = ["LOC:other", "LOC:mount"]

        elif "WORK_OF_ART" == ent_label:
            new_ent_label = ["ENTY:cremat"]

        # this mapping is incorrect as ENTY should be separate
        elif "MONEY" == ent_label:
            new_ent_label = ["ENTY:currency", "NUM:money"]

        # THERE IS NO ROTH TAG FOR TIME
        elif "DATE" == ent_label or "TIME" == ent_label:
            new_ent_label = ["NUM:date", "NUM:period"]

        elif "EVENT" == ent_label:
            new_ent_label = ["ENTY:event"]

        elif "PRODUCT" == ent_label:
            # ignore 'ENTY:other'
            new_ent_label = ["ENTY:product", "ENTY:food", "ENTY:veh"]

        elif "CARDINAL" == ent_label:
            new_ent_label = ["NUM:count", "NUM:code"]

        elif "PERSON" == ent_label:
            new_ent_label = ["HUM:ind"]

        elif "ORG" == ent_label:
            new_ent_label = ["HUM:gr"]
        # check this mapping
        elif "NORP" == ent_label:
            new_ent_label = ["ENTY:religion", "HUM:gr"]

        elif "QUANTITY" == ent_label:
            new_ent_label = ["NUM:volsize", "NUM:dist", "NUM:temp", "NUM:weight"]

        elif "LANGUAGE" == ent_label:
            new_ent_label = ["ENTY:lang"]

        elif "PERCENT" == ent_label:
            new_ent_label = ["NUM:perc"]

        elif "MEASUREMENT" == ent_label:
            new_ent_label = ["NUM:speed"]

        return new_ent_label

    def convert_medical_label(self, ent_label):
        # non comforming Spacy -> Roth will be OTHERS
        # skip LAW, it only maps to 'ENTY:other'

        dictionary = {
            "GGP": "GENE_OR_GENE_PRODUCT",
            "CL": "CELL",
        }
        if ent_label in dictionary:
            return dictionary[ent_label]
        else:
            return ent_label

    def get_pure_rake(self, doc):
        phrases = []
        r = Rake(
            min_length=1,
            max_length=4,
            ranking_metric=Metric.WORD_DEGREE,
            punctuations=[".", ",", "%", "$"],
        )
        r.extract_keywords_from_text(doc.text)
        keywords = r.get_ranked_phrases_with_scores()
        for kw in keywords:
            if kw[0] > 2:
                phrases.append(self.format_phrase(kw[1]))
        return phrases

    def get_pure_yake(self, doc):
        phrases = []
        y = yake.KeywordExtractor(dedupLim=0.5)
        keywords = y.extract_keywords(doc.text)
        for kw in keywords:
            if kw[1] < 0.02:
                phrases.append(self.format_phrase(kw[0]))
        return phrases

    def get_combination(self, doc):
        punct_count = 0
        for token in doc:
            punct_count += 1 if token.is_punct else 0
        word_count = len(doc) - punct_count
        # print(doc.ents)
        y_phrases = self.get_pure_yake(doc) if word_count > 5 else []
        r_phrases = self.get_pure_rake(doc)
        f_phrases = self.get_filtered_phrases(doc)
        combined_phrases = list(set(r_phrases + y_phrases))
        for phrase in f_phrases:
            if phrase.lower() in combined_phrases:
                combined_phrases.remove(phrase.lower())
            combined_phrases.append(phrase)

        # for token in doc:
        #     if token.pos_ in ["VERB", "ADV"]:
        #         print(token.text)

        return list(set(combined_phrases))

    def get_filtered_phrases(self, doc):
        phrases = []
        for chunk in doc.noun_chunks:
            if chunk.root.pos_ not in ["DET", "PRON", "PUNCT", "NUM", "PART"]:
                p = " ".join(
                    [
                        token.text
                        for token in chunk
                        if token.pos_ not in ["DET", "PRON", "PUNCT", "NUM", "PART"]
                    ],
                )
                phrases.append(self.format_phrase(p))
        return phrases

    def format_phrase(self, phrase):
        if " - " in phrase:
            phrase = phrase.replace(" - ", "-")
        if " 's" in phrase:
            phrase = phrase.replace(" 's", "'s")
        if " ’s" in phrase:
            phrase = phrase.replace(" ’s", "'s")
        return phrase

    def get_main_subject(self, doc) -> List[str]:

        """
        Takes a Doc as input
        Returns a list of strings representing the key entity
        If multiple references exist, returns the most canonical
        mention of a referenced entity
        """
        # TODO: Coref should really be a preprocess step
        # before the rest executes; but .coref_resolved
        # returns a unicode string, so need to think how
        # to do it
        # TODO: Can do better than return unchanged text
        # when no "good" answer is found

        # if doc._.has_coref:
        #     main = list(
        #         cluster.main.text
        #         for cluster in doc._.coref_clusters
        #         # if cluster.main in doc.ents
        #     )

        #     return main[0]
        if doc.ents:
            candidates = []
            root = [token for token in doc if token.head == token][0]
            for ent in doc.ents:
                if ent.label_ in [
                    "CARDINAL",
                    "FAC",
                    "LOC",
                    "MONEY",
                    "ORG",
                    "GPE",
                    "QUANTITY",
                ]:
                    # if ent.root.dep_ in ["nsubj","dobj","pobj","nsubjpass"]:
                    if root.is_ancestor(ent.root):
                        candidates.append((ent.text, len(list(ent.root.ancestors))))

            # TODO: when we can classify question based on topic,
            # return candidate with appropriate ent type instead
            # of just the first one
            if len(candidates) > 0:
                return min(candidates)[0]
            else:
                return doc.text
        else:
            tok = None
            for token in doc:
                if token.dep_ in ["nsubj", "dobj", "nsubjpass", "pobj"]:
                    tok = token.text
            if tok:
                return tok
            else:
                return [token for token in doc if token.head == token][0]

    def parse_deps(self, doc, keyword):
        doc_spans = []
        for sent in doc.sents:
            if keyword in sent.text:
                for word in sent:
                    if word.dep_ in ("nsubj", "aux", "xcomp", "ccomp", "nsubjpass"):
                        subtree_span = doc[word.left_edge.i : word.right_edge.i + 1]
                        doc_spans.append((subtree_span.root, subtree_span))
        return doc_spans

    def filter_doc_spans(self, doc_spans):
        for doc in doc_spans:
            for w in [t for t in doc[1].noun_chunks]:
                print(f"w: {w[0]} | {w[0].dep_} | {w[0].like_num} | {w[0].ent_type_}")

            res = list(
                filter(
                    lambda w: not w[0].like_num
                    and not w[0].is_punct
                    and not w[0].dep_ == "nummod"
                    and not w[0].ent_type_ == "PERCENT",
                    [t for t in doc[1].noun_chunks],
                ),
            )
            return res

    def extract_keywords(self, doc, keywords):
        keyword_matches = []
        for keyword in keywords:
            if len(keyword.split()) > 1:
                keyword = [t.text for t in self.nlp(keyword) if t.dep_ == "ROOT"][0]
            kw = []
            doc_spans = self.parse_deps(doc, keyword)
            hits = self.filter_doc_spans(doc_spans)
            if hits is not None:
                kw.extend([x.text for x in hits if x is not None])
                print(f"kw: {kw}")
            if len(kw) > 0:
                keyword_matches.append(kw)

        return keyword_matches

    def get_resolved_text(self, doc) -> str:
        """
        Takes a Doc as input
        Returns a string with coreferences
        replaced by the canonical referent
        """
        # if doc._.has_coref:
        #     return doc._.coref_resolved
        # else:
        #     return doc.text
        return doc.text

    def extract_phrases(self, doc) -> List[str]:
        """
        Takes a Doc as input
        Returns lists of tokens as strings
        joined tokens as strings according
        to their dependency parse info.

        TODO: Verify return type (str vs Token)
        TODO: Verify objective of functionality
        """

        visited = set()
        phrases = []
        phrase_texts = []
        for token in doc:
            if (
                token.n_lefts + token.n_rights > 0
                and not token.is_punct
                and not token.is_stop
            ):
                phrase_list = self.c_phrases(token, visited)
                phrase_texts.extend(
                    [" ".join([t.text for t in phrase]) for phrase in phrase_list],
                )
                phrases.extend(phrase_list)

        return phrase_texts

    def c_phrases(self, token, visited):
        if token is None or token.i in visited:
            return []
        visited.add(token.i)
        l_phrases = []
        for t in token.lefts:
            if not t.is_punct:
                collected = self.c_phrases(t, visited)
                l_phrases.extend(collected)

        r_phrases = []
        for t in token.rights:
            if not t.is_punct:
                collected = self.c_phrases(t, visited)
                r_phrases.extend(collected)

        # try to fit the token into left or right phrases

        if len(l_phrases) > 0:
            # fit into left phrases
            l_phrases[-1].append(token)
        elif len(r_phrases) > 0:
            # fit into right phrases
            r_phrases[0].insert(0, token)
        else:
            l_phrases.append([token])

        l_phrases.extend(r_phrases)

        return self.merge_tags(l_phrases)

    def merge_tags(self, tags):
        if len(tags) < 1:
            return []
        tag, rem = tags[0], tags[1:]
        rem_tags = self.merge_tags(rem)
        if len(rem_tags) > 0 and len(rem_tags[0]) == 1:
            tag.extend(rem_tags[0])
            return [tag]
        elif len(tag) == 1 and len(rem_tags) > 0:
            rem_tags[0].insert(0, tag[0])
            return rem_tags
        else:
            return [tag] + rem_tags

    def post(self):
        """
        Handles post requests
        Returns jsonified response text
        """
        
        args = self.req_parser.parse_args()
        
        try:

            wall_time = default_timer()
            args = self.req_parser.parse_args()
            if "option" not in args.keys() or "text" not in args.keys():
                return make_response(
                    jsonify(
                        {
                            "status": "fail",
                            "message": '"option" and "text" are required arguments',
                        },
                    ),
                    400,
                )

            option = args["option"]
            if option not in PERMITTED_OPERATIONS:
                msg = {
                    "status": "fail",
                    "message": "Phraser has no option " + str(option),
                }
                return make_response(jsonify(msg), 400)

            else:
                texts = args["texts"]
                domain = args.get("domain", "general")

                self.logger.info(
                    f"Running NLP server with option {option} on {len(texts)} samples",
                )

                docs = self.nlp.pipe(texts)

                if option == "get_main_subject":
                    msg = {
                        "data": [doc._.get_main_subject() for doc in docs],
                    }
                elif option == "get_resolved_text":
                    msg = {
                        "data": [doc._.get_resolved_text() for doc in docs],
                    }
                elif option == "extract_phrases":
                    msg = {
                        "data": [doc._.extract_phrases() for doc in docs],
                    }
                elif option == "get_filtered_phrases":
                    msg = {
                        "data": [doc._.get_filtered_phrases() for doc in docs],
                    }
                elif option == "extract_keywords":
                    keywords = list(args["keywords"])
                    msg = {
                        "data": [
                            doc._.extract_keywords(keywords=keywords) for doc in docs
                        ],
                    }
                elif option == "get_pure_rake":
                    msg = {
                        "data": [doc._.get_pure_rake() for doc in docs],
                    }
                elif option == "get_pure_yake":
                    msg = {
                        "data": [doc._.get_pure_yake() for doc in docs],
                    }
                elif option == "get_combination":
                    msg = {
                        "data": [doc._.get_combination() for doc in docs],
                    }
                elif option == "get_doc_ents":
                    msg = {
                        "data": self.get_doc_ents(texts, domain=domain),
                    }
                elif option == "get_doc_tags":
                    msg = {
                        "data": self.get_doc_tags(texts),
                    }
                elif option == "convert_question_to_sentence":
                    msg = {
                        "data": self.convert_question_to_sentence(texts),
                    }
                else:
                    make_response(jsonify({"msg": "unknown option {option}"}), 404)

            wall_time = (default_timer() - wall_time) * 1000
            self.logger.info(
                f"Running NLP server with option {option} on {len(texts)} samples finished in {wall_time:.2f}ms, "
                f"{wall_time/len(texts):.2f}ms per sample",
            )
            return make_response(jsonify(msg), 200)

        except Exception as e:
            msg = {"status": "fail", "message": str(e)}
            self.logger.error(str(e))
            return make_response(jsonify(msg), 500)


if __name__ == "__main__":
    pass
