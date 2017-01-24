"""
CYK Parser
@author: Will Kearns
"""
import sys
import nltk
import pandas as pd


def iterable_df(size):
    """
    Initializes a symmetric Pandas DataFrame of lists.
    :param size: the dimension of the symmetric matrix.
    :return: Pandas DataFrame
    """
    df = pd.DataFrame(columns=range(size), index=range(size))
    return df.applymap(lambda x: [])


class Node:

    def __init__(self, _id, left, right):
        """
        A vertex of the parse tree

        :param _id: Identifier, in the case of CYK is a Non-terminal
        :param left: Child of left branch of tree w.r.t. this node
        :param right: Child of right branch of tree w.r.t. this node
        """
        self.id = _id
        self.left = left
        self.right = right

    def print(self):
        sys.stdout.write("(" + self.id + " ")


class CYK:

    def __init__(self):
        self.table = None
        self.shadow_matrix = None
        self.tokens = None

    def parse(self, sentence, grammar):
        """
        Fills the CYK table with nonterminal tags corresponding to word spans.

        :param sentence: String to be parsed by the grammar.
        :param grammar: Context-free Grammar as NLTK CFG Object.
        :return: CYK table
        """
        self.tokens = nltk.word_tokenize(sentence)
        self.table = iterable_df(len(self.tokens))
        self.shadow_matrix = iterable_df(len(self.tokens))

        for i, word in enumerate(self.tokens):
            # Initial Non-terminals are assigned based on Lexical rules
            for rule in grammar.productions():
                if rule.is_lexical() and rule.rhs()[0] == word:
                    self.table.iloc[i][i].append(str(rule.lhs()))
            # Merge
            j = i - 1
            while j >= 0:
                k = j + 1
                while k <= i:
                    non_lexical_match = []
                    pointers = []
                    for B in self.table.iloc[j][k-1]:
                        for C in self.table.iloc[k][i]:
                            for rule in grammar.productions():
                                if str(rule.rhs()[0]) == B and str(rule.rhs()[1]) == C:
                                    non_lexical_match.append(str(rule.lhs()))
                                    pointers.append(((j, k-1, B), (k, i, C)))
                    for element in self.table.iloc[j][i]:
                        non_lexical_match.append(element)
                    for element in self.shadow_matrix.iloc[j][i]:
                        pointers.append(element)
                    self.table.iloc[j][i] = non_lexical_match
                    self.shadow_matrix.iloc[j][i] = pointers
                    k += 1
                j -= 1

        return self.table

    def spawn_nodes(self, row, column, nonterminal, index):
        """
        Creates a node and recursively generates its children.

        :param row: row of parent node in table.
        :param column: column of parent node in table.
        :param nonterminal: id of node
        :param index: index of target nonterminal
        :return:
        """
        pointers = self.shadow_matrix.iloc[row][column]
        if not pointers:
            sys.stdout.write("(" + nonterminal + " ")
            sys.stdout.write(self.tokens[column] + ") ")
            return None
        else:
            sys.stdout.write("(" + nonterminal + " ")
            # Spawn left child
            B = pointers[index][0]
            idx = self.table.iloc[B[0]][B[1]].index(B[2])
            left = self.spawn_nodes(B[0], B[1], B[2], idx)

            # Spawn right child
            C = pointers[index][1]
            idx = self.table.iloc[C[0]][C[1]].index(C[2])
            right = self.spawn_nodes(C[0], C[1], C[2], idx)

            if not left or not right:
                sys.stdout.write(") ")
        return Node(nonterminal, left, right)

    def decode(self, table):
        rows, cols = table.shape
        i, j = 0, cols - 1
        parses = []
        for idx, nonterminal in enumerate(table.iloc[i][j]):
            tree = self.spawn_nodes(i, j, nonterminal, idx)
            parses.append(tree)

if __name__ == "__main__":
    grammar_file = open(sys.argv[1], "r")
    cfg = nltk.CFG.fromstring(grammar_file.read())
    grammar_file.close()
    print(cfg.productions())

    sentence_file = open(sys.argv[2], "r")

    for line in sentence_file:
        print(line)
        cyk = CYK()
        parse_table = cyk.parse(line, cfg)
        print(parse_table)
        cyk.decode(parse_table)
