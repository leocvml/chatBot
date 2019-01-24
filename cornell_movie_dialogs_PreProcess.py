import random

''' 
    1. Read from 'movie-lines.txt'
    2. Create a dictionary with ( key = line_id, value = text )
'''


def get_id2line():
    lines = open('movie_lines.txt',encoding='cp1252').read().split('\n')
    id2line = {}
    for line in lines:
        _line = line.split(' +++$+++ ')
        if len(_line) == 5:
            id2line[_line[0]] = _line[4]
    return id2line


'''
    1. Read from 'movie_conversations.txt'
    2. Create a list of [list of line_id's]
'''


def get_conversations():
    conv_lines = open('movie_conversations.txt').read().split('\n')
    convs = []
    for line in conv_lines[:-1]:
        _line = line.split(' +++$+++ ')[-1][1:-1].replace("'", "").replace(" ", "")
        convs.append(_line.split(','))
    return convs


'''
    1. Get each conversation
    2. Get each line from conversation
    3. Save each conversation to file
'''


def extract_conversations(convs, id2line, path=''):
    idx = 0
    for conv in convs:
        f_conv = open(path + str(idx) + '.txt', 'w')
        for line_id in conv:
            f_conv.write(id2line[line_id])
            f_conv.write('\n')
        f_conv.close()
        idx += 1


'''
    Get lists of all conversations as Questions and Answers
    1. [questions]
    2. [answers]
'''


def gather_dataset(convs, id2line):
    questions = [];
    answers = []

    for conv in convs:
        if len(conv) % 2 != 0:
            conv = conv[:-1]
        for i in range(len(conv)):
            if i % 2 == 0:
                questions.append(id2line[conv[i]])
            else:
                answers.append(id2line[conv[i]])

    return questions, answers


'''
    We need 4 files
    1. train.enc : Encoder input for training
    2. train.dec : Decoder input for training
    3. test.enc  : Encoder input for testing
    4. test.dec  : Decoder input for testing
'''


def prepare_seq2seq_files(questions, answers, path='', TESTSET_SIZE=30000):
    # open files
    train_enc = open(path + 'train.txt', 'w')
    train_dec = open(path + 'train.txt', 'w')
    test_enc = open(path + 'test.txt', 'w')
    test_dec = open(path + 'test.txt', 'w')

    # train_enc = open(path + 'train.enc', 'w')
    # train_dec = open(path + 'train.dec', 'w')
    # test_enc = open(path + 'test.enc', 'w')
    # test_dec = open(path + 'test.dec', 'w')

    # choose 30,000 (TESTSET_SIZE) items to put into testset
    test_ids = random.sample([i for i in range(len(questions))], TESTSET_SIZE)

    for i in range(len(questions)):
        if i in test_ids:
            test_enc.write(questions[i] + '\n')
            test_dec.write(answers[i] + '\n')
        else:
            train_enc.write(questions[i] + '\n')
            train_dec.write(answers[i] + '\n')
        if i % 10000 == 0:
            print('\n>> written %d lines' % (i))

            # close files
    train_enc.close()
    train_dec.close()
    test_enc.close()
    test_dec.close()


####
# main()
####

id2line = get_id2line()
print('>> gathered id2line dictionary.\n')
convs = get_conversations()
print('>> gathered conversations.\n')
questions, answers = gather_dataset(convs, id2line)
print(len(questions ))
print(len(answers))
index = 6541
print(questions[index])
print(answers[index])

# q_e = []
# a_e = []
# with open('question.txt', 'w') as f:
#     for index in range(len(questions)):
#         try:
#
#             f.write(questions[index] + '\n')
#         except Exception:
#             q_e.append(index)
#             pass
#
# with open('answer.txt','w') as f:
#     for index in range(len(answers)):
#         try:
#             f.write(answers[index] + '\n')
#         except Exception:
#             a_e.append(index)
#             pass
#
# print(len(q_e))
# print(len(a_e))

q_w = open('question.txt','w')
a_w = open('answer.txt','w')

correct = []
for index in range(len(answers)):
    try:
        q_w.write(questions[index] + '\t')
        q_w.write(answers[index] + '\n')
        correct.append(index)
    except Exception:
        pass

print(len(correct))

t_s = open('trainingset.txt','w')
for index in correct:
    try:
        t_s.write(questions[index] + '@')
        t_s.write(answers[index] + '\n')
    except Exception:
        pass
t_s.close()

with open('trainingset.txt','r') as f:
    all = f.readlines()

print(len(all))

for each in all :
    q,a = each.split('@')
    print(q,a)

# print ('>> gathered questions and answers.')
# prepare_seq2seq_files(questions,answers)