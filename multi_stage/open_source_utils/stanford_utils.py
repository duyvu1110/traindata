import copy
from stanfordcorenlp import StanfordCoreNLP
import stanza


class stanfordFeature(object):
    def __init__(self, sentences, stanford_path):
        """
        :param sentences: a list of sentence, [sentence1, sentence2......]
        :param stanford_path: nlp stanford core path
        :param lang: denote which language sentences need to process
        """
        self.sentences_col = sentences
        stanza.download('vi')
        self.nlp = stanza.Pipeline('vi', processors='tokenize,pos,lemma,depparse', verbose=False)

        # using set to store label type
        self.pos_dict, self.pos_index = {"PAD": 0}, 1
        self.dep_label_dict, self.dep_label_index = {}, 1
        self.vocab = {}

        # store the maximum length of sequence
        self.max_len = -1

    def get_tokenizer(self):
        """
        :return: a list of token by stanford tokenizer
        """
        input_tokens = []
        for i in range(len(self.sentences_col)):
            # token_list = self.nlp.word_tokenize(self.sentences_col[i])
            token_list = []
            doc = self.nlp(self.sentences_col[i])
            for sentence in doc.sentences:
                for token in sentence.tokens:
                    token_list.append(token.text)
            # token_list = self.nlp(self.sentences_col[i])
            input_tokens.append(token_list)

        return input_tokens

    def get_pos_feature(self, pos_dict, pos_index):
        """
        :param: pos_dict:
        :param: pos_index:
        :return: a list of pos-tag, with id
        """
        self.pos_dict = copy.deepcopy(pos_dict)
        self.pos_index = pos_index

        pos_feature = []
        for index in range(len(self.sentences_col)):
            # tag_list = self.nlp.pos_tag(self.sentences_col[index])
            # pos_tag_list = list(list(zip(*tag_list))[1])
            pos_tag_list = []
            doc = self.nlp(self.sentences_col[index])
            for sentence in doc.sentences:
                for word in sentence.words:
                    pos_tag_list.append(word.pos)

            # update pos-tag set
            for tag in pos_tag_list:
                if tag not in self.pos_dict:
                    self.pos_dict[tag] = self.pos_index
                    self.pos_index += 1

            pos_feature.append([self.pos_dict[tag] for tag in pos_tag_list])

        return pos_feature, self.pos_dict, self.pos_index

    def get_dep_feature(self):
        """
        :return: dependency matrix and dependency label matrix
        """
        dep_matrix_feature, dep_label_feature = [], []
        for index in range(len(self.sentences_col)):
            # dep_parse = self.nlp.dependency_parse(self.sentences_col[index])
            # label_col = list(list(zip(*dep_parse))[0])
            # out_node, in_node = list(list(zip(*dep_parse))[1]), list(list(zip(*dep_parse))[2])
            doc = self.nlp(self.sentences_col[index])
            dep_parse = [(word.deprel, word.head, word.text) for sent in doc.sentences for word in sent.words]
            
            label_col = [dep[0] for dep in dep_parse]
            out_node = [dep[1] for dep in dep_parse]
            in_node = [dep[2] for dep in dep_parse]

            # define dep matrix and dep label matrix
            dep_matrix = [[0 for _ in range(len(out_node))] for j in range(len(out_node))]
            dep_label_matrix = copy.deepcopy(dep_matrix)

            # self loop
            for i in range(len(out_node)):
                dep_matrix[i][i] = 1

            # get dep_matrix and dep_label_matrix
            for i in range(len(out_node)):
                if out_node[i] == 0:
                    continue

                dep_matrix[out_node[i] - 1][in_node[i] - 1] = 1
                dep_matrix[in_node[i] - 1][out_node[i] - 1] = 1

                if label_col[i] not in self.dep_label_dict:
                    self.dep_label_dict[label_col[i]] = self.dep_label_index
                    self.dep_label_index = self.dep_label_index + 1

                dep_label_matrix[out_node[i] - 1][in_node[i] - 1] = self.dep_label_dict[label_col[i]]

            dep_matrix_feature.append(dep_matrix)
            dep_label_feature.append(dep_label_matrix)

        return dep_matrix_feature, dep_label_feature
