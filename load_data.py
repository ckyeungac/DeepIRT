import numpy as np 
from utils import getLogger

class DataLoader():
    def __init__(self, n_questions, seq_len, separate_char):
        self.separate_char = separate_char
        self.n_questions = n_questions
        self.seq_len = seq_len
    
    def load_data(self, path):
        q_data = []
        qa_data = []
        with open(path, 'r') as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                # skip the number of sequence
                if line_idx%3 == 0:
                    continue
                # handle question_line
                elif line_idx%3 == 1:
                    q_tag_list = line.split(self.separate_char)
                # handle answer-line
                elif line_idx%3 == 2:
                    a_tag_list = line.split(self.separate_char)

                    # find the number of split for this sequence
                    n_split = len(q_tag_list) // self.seq_len
                    if len(q_tag_list) % self.seq_len != 0:
                        n_split += 1
                    
                    for k in range(n_split):
                        # temporary container for each sequence
                        q_container = list()
                        qa_container = list()

                        start_idx = k*self.seq_len
                        end_idx = min((k+1)*self.seq_len, len(a_tag_list))

                        for i in range(start_idx, end_idx):
                            q_value = int(q_tag_list[i])
                            a_value = int(a_tag_list[i])  # either be 0 or 1
                            qa_value = q_value + a_value * self.n_questions
                            q_container.append(q_value)
                            qa_container.append(qa_value)
                        q_data.append(q_container)
                        qa_data.append(qa_container)

        # convert it to numpy array
        q_data_array = np.zeros((len(q_data), self.seq_len))
        qa_data_array = np.zeros((len(q_data), self.seq_len))
        for i in range(len(q_data)):
            _q_data = q_data[i]
            _qa_data = qa_data[i]
            q_data_array[i, :len(_q_data)] = _q_data
            qa_data_array[i, :len(_qa_data)] = _qa_data

        return q_data_array, qa_data_array
        