"""

Main module. From: https://github.com/LIAAD/yake.

In-depth journal paper at Information Sciences Journal
Campos, R., Mangaravite, V., Pasquali, A., Jatowt, A., Jorge, A., Nunes, C.
and Jatowt, A. (2020). YAKE! Keyword Extraction from Single Documents using
Multiple Local Features. In Information Sciences Journal. Elsevier, Vol 509,
pp 257-289. pdf

ECIR'18 Best Short Paper
Campos R., Mangaravite V., Pasquali A., Jorge A.M., Nunes C., and Jatowt A.
(2018). A Text Feature Based Automatic Keyword Extraction Method for Single
Documents. In: Pasi G., Piwowarski B., Azzopardi L., Hanbury A. (eds).
Advances in Information Retrieval. ECIR 2018 (Grenoble, France. March 26 – 29).
Lecture Notes in Computer Science, vol 10772, pp. 684 - 691. pdf

Campos R., Mangaravite V., Pasquali A., Jorge A.M., Nunes C., and Jatowt A.
(2018). YAKE! Collection-independent Automatic Keyword Extractor. In:
Pasi G., Piwowarski B., Azzopardi L., Hanbury A. (eds). Advances in Information
Retrieval. ECIR 2018 (Grenoble, France. March 26 – 29). Lecture Notes in
Computer Science, vol 10772, pp. 806 - 810. pdf

"""
import os

import jellyfish

from .datarepresentation import DataCore
from .Levenshtein import Levenshtein


class KeywordExtractor:
    def __init__(
        self,
        lan="en",
        n=3,
        dedupLim=0.9,
        dedupFunc="seqm",
        windowsSize=1,
        top=20,
        features=None,
        stopwords=None,
    ):
        self.lan = lan

        dir_path = os.path.dirname(os.path.realpath(__file__))

        local_path = os.path.join("StopwordsList", "stopwords_%s.txt" % lan[:2].lower())

        if not os.path.exists(os.path.join(dir_path, local_path)):
            local_path = os.path.join("StopwordsList", "stopwords_noLang.txt")

        resource_path = os.path.join(dir_path, local_path)

        if stopwords is None:
            try:
                with open(resource_path, encoding="utf-8") as stop_fil:
                    self.stopword_set = set(stop_fil.read().lower().split("\n"))
            except Exception:
                print("Warning, read stopword list as ISO-8859-1")
                with open(resource_path, encoding="ISO-8859-1") as stop_fil:
                    self.stopword_set = set(stop_fil.read().lower().split("\n"))
        else:
            self.stopword_set = set(stopwords)

        self.n = n
        self.top = top
        self.dedupLim = dedupLim
        self.features = features
        self.windowsSize = windowsSize
        if dedupFunc == "jaro_winkler" or dedupFunc == "jaro":
            self.dedu_function = self.jaro
        elif dedupFunc.lower() == "sequencematcher" or dedupFunc.lower() == "seqm":
            self.dedu_function = self.seqm
        else:
            self.dedu_function = self.levs

    def jaro(self, cand1, cand2):
        return jellyfish.jaro_winkler(cand1, cand2)

    def levs(self, cand1, cand2):
        return 1.0 - jellyfish.levenshtein_distance(cand1, cand2) / max(
            len(cand1), len(cand2),
        )

    def seqm(self, cand1, cand2):
        return Levenshtein.ratio(cand1, cand2)

    def extract_keywords(self, text):
        text = text.replace("\n\t", " ")
        dc = DataCore(
            text=text,
            stopword_set=self.stopword_set,
            windowsSize=self.windowsSize,
            n=self.n,
        )
        dc.build_single_terms_features(features=self.features)
        dc.build_mult_terms_features(features=self.features)
        resultSet = []
        todedup = sorted(
            [cc for cc in dc.candidates.values() if cc.isValid()], key=lambda c: c.H,
        )

        if self.dedupLim >= 1.0:
            return ([(cand.H, cand.unique_kw) for cand in todedup])[: self.top]

        for cand in todedup:
            toadd = True
            for (h, candResult) in resultSet:
                dist = self.dedu_function(cand.unique_kw, candResult.unique_kw)
                if dist > self.dedupLim:
                    toadd = False
                    break
            if toadd:
                resultSet.append((cand.H, cand))
            if len(resultSet) == self.top:
                break

        return [(cand.unique_kw, h) for (h, cand) in resultSet]
