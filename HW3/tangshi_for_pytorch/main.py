import numpy as np
import collections
import os
import torch
from torch.autograd import Variable
import torch.optim as optim

import rnn

start_token = 'G'
end_token = 'E'
batch_size = 64

EMBEDDING_DIM = 100
HIDDEN_DIM = 64
TRAIN_BATCH_SIZE = 100
EPOCHS = 30

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
POEMS_PATH = os.path.join(BASE_DIR, 'poems.txt')
MODEL_PATH = os.path.join(BASE_DIR, 'poem_generator_rnn.pt')
LEGACY_MODEL_PATH = os.path.join(BASE_DIR, 'poem_generator_rnn')


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def process_poems1(file_name):
    """

    :param file_name:
    :return: poems_vector  have tow dimmention ,first is the poem, the second is the word_index
    e.g. [[1,2,3,4,5,6,7,8,9,10],[9,6,3,8,5,2,7,4,1]]

    """
    poems = []
    with open(file_name, "r", encoding='utf-8', ) as f:
        for line in f.readlines():
            try:
                title, content = line.strip().split(':', 1)
                # content = content.replace(' ', '').replace('，','').replace('。','')
                content = content.replace(' ', '')
                if '_' in content or '(' in content or '（' in content or '《' in content or '[' in content or \
                                start_token in content or end_token in content:
                    continue
                if len(content) < 5 or len(content) > 80:
                    continue
                content = start_token + content + end_token
                poems.append(content)
            except ValueError as e:
                pass
    # 按诗的字数排序
    poems = sorted(poems, key=lambda line: len(line))
    # print(poems)
    # 统计每个字出现次数
    all_words = []
    for poem in poems:
        all_words += [word for word in poem]
    counter = collections.Counter(all_words)  # 统计词和词频。
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])  # 排序
    words, _ = zip(*count_pairs)
    words = words[:len(words)] + (' ',)
    word_int_map = dict(zip(words, range(len(words))))
    poems_vector = [list(map(word_int_map.get, poem)) for poem in poems]
    return poems_vector, word_int_map, words

def process_poems2(file_name):
    """
    :param file_name:
    :return: poems_vector  have tow dimmention ,first is the poem, the second is the word_index
    e.g. [[1,2,3,4,5,6,7,8,9,10],[9,6,3,8,5,2,7,4,1]]

    """
    poems = []
    with open(file_name, "r", encoding='utf-8', ) as f:
        # content = ''
        for line in f.readlines():
            try:
                line = line.strip()
                if line:
                    content = line.replace(' '' ', '').replace('，','').replace('。','')
                    if '_' in content or '(' in content or '（' in content or '《' in content or '[' in content or \
                                    start_token in content or end_token in content:
                        continue
                    if len(content) < 5 or len(content) > 80:
                        continue
                    # print(content)
                    content = start_token + content + end_token
                    poems.append(content)
                    # content = ''
            except ValueError as e:
                # print("error")
                pass
    # 按诗的字数排序
    poems = sorted(poems, key=lambda line: len(line))
    # print(poems)
    # 统计每个字出现次数
    all_words = []
    for poem in poems:
        all_words += [word for word in poem]
    counter = collections.Counter(all_words)  # 统计词和词频。
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])  # 排序
    words, _ = zip(*count_pairs)
    words = words[:len(words)] + (' ',)
    word_int_map = dict(zip(words, range(len(words))))
    poems_vector = [list(map(word_int_map.get, poem)) for poem in poems]
    return poems_vector, word_int_map, words

def generate_batch(batch_size, poems_vec, word_to_int):
    n_chunk = len(poems_vec) // batch_size
    x_batches = []
    y_batches = []
    for i in range(n_chunk):
        start_index = i * batch_size
        end_index = start_index + batch_size
        x_data = poems_vec[start_index:end_index]
        y_data = []
        for row in x_data:
            y  = row[1:]
            y.append(row[-1])
            y_data.append(y)
        """
        x_data             y_data
        [6,2,4,6,9]       [2,4,6,9,9]
        [1,4,2,8,5]       [4,2,8,5,5]
        """
        # print(x_data[0])
        # print(y_data[0])
        # exit(0)
        x_batches.append(x_data)
        y_batches.append(y_data)
    return x_batches, y_batches


def build_model(word_to_int, batch_sz, device):
    vocab_len = len(word_to_int) + 1
    word_embedding = rnn.word_embedding(vocab_length=vocab_len, embedding_dim=EMBEDDING_DIM)
    model = rnn.RNN_model(
        batch_sz=batch_sz,
        vocab_len=vocab_len,
        word_embedding=word_embedding,
        embedding_dim=EMBEDDING_DIM,
        lstm_hidden_dim=HIDDEN_DIM,
    )
    model.to(device)
    return model


def save_checkpoint(model, word_to_int, vocabularies, model_path=MODEL_PATH):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'word_to_int': word_to_int,
        'vocabularies': list(vocabularies),
        'config': {
            'embedding_dim': EMBEDDING_DIM,
            'hidden_dim': HIDDEN_DIM,
        },
    }
    torch.save(checkpoint, model_path)


def load_model_and_vocab(model_path=None):
    device = get_device()
    if model_path is None:
        if os.path.exists(MODEL_PATH):
            model_path = MODEL_PATH
        else:
            model_path = LEGACY_MODEL_PATH
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # New format: checkpoint with both weights and vocabulary mapping.
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint and 'word_to_int' in checkpoint:
        word_to_int = checkpoint['word_to_int']
        vocabularies = tuple(checkpoint['vocabularies'])
        model = build_model(word_to_int, batch_sz=batch_size, device=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model, word_to_int, vocabularies, device

    # Legacy format: weights only. Requires current data preprocessing to match training preprocessing.
    _, word_to_int, vocabularies = process_poems1(POEMS_PATH)
    model = build_model(word_to_int, batch_sz=batch_size, device=device)
    try:
        model.load_state_dict(checkpoint)
    except RuntimeError as e:
        raise RuntimeError(
            '旧模型文件只包含权重，且当前词表与训练时不一致。\n'
            '请先执行 run_training() 重新训练并保存新格式 checkpoint（包含词表映射），'
            '然后再进行生成。'
        ) from e
    model.eval()
    return model, word_to_int, vocabularies, device


def run_training():
    # 处理数据集
    # poems_vector, word_to_int, vocabularies = process_poems2('./tangshi.txt')
    poems_vector, word_to_int, vocabularies = process_poems1(POEMS_PATH)
    # 生成batch
    print("finish  loadding data")
    BATCH_SIZE = TRAIN_BATCH_SIZE
    device = get_device()
    print('using device:', device)
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    torch.manual_seed(5)
    rnn_model = build_model(word_to_int, batch_sz=BATCH_SIZE, device=device)

    # optimizer = optim.Adam(rnn_model.parameters(), lr= 0.001)
    optimizer=optim.RMSprop(rnn_model.parameters(), lr=0.01)

    loss_fun = torch.nn.NLLLoss()
    # rnn_model.load_state_dict(torch.load('./poem_generator_rnn.pt'))  # if you have already trained your model you can load it by this line.

    for epoch in range(EPOCHS):
        batches_inputs, batches_outputs = generate_batch(BATCH_SIZE, poems_vector, word_to_int)
        n_chunk = len(batches_inputs)
        for batch in range(n_chunk):
            batch_x = batches_inputs[batch]
            batch_y = batches_outputs[batch] # (batch , time_step)

            loss = torch.tensor(0.0, device=device)
            for index in range(BATCH_SIZE):
                x = np.array(batch_x[index], dtype = np.int64)
                y = np.array(batch_y[index], dtype = np.int64)
                x = Variable(torch.from_numpy(np.expand_dims(x,axis=1))).to(device)
                y = Variable(torch.from_numpy(y )).to(device)
                pre = rnn_model(x)
                loss += loss_fun(pre , y)
                if index == 0:
                    _, pre = torch.max(pre, dim=1)
                    print('prediction', pre.data.tolist()) # the following  three line can print the output and the prediction
                    print('b_y       ', y.data.tolist())   # And you need to take a screenshot and then past is to your homework paper.
                    print('*' * 30)
            loss  = loss  / BATCH_SIZE
            print("epoch  ",epoch,'batch number',batch,"loss is: ", loss.item())
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(rnn_model.parameters(), 1)
            optimizer.step()

            if batch % 20 ==0:
                save_checkpoint(rnn_model, word_to_int, vocabularies, MODEL_PATH)
                print("finish  save model")



def to_word(predict, vocabs):  # 预测的结果转化成汉字
    sample = np.argmax(predict)

    if sample >= len(vocabs):
        sample = len(vocabs) - 1

    return vocabs[sample]


def pretty_print_poem(poem):  # 令打印的结果更工整
    shige=[]
    for w in poem:
        if w == start_token or w == end_token:
            break
        shige.append(w)
    poem_sentences = poem.split('。')
    for s in poem_sentences:
        if s != '' and len(s) > 10:
            print(s + '。')


def gen_poem(begin_word):

    rnn_model, word_int_map, vocabularies, device = load_model_and_vocab()

    # 指定开始的字
    if begin_word not in word_int_map:
        raise ValueError('输入的起始字不在词表中，请更换起始字或重新训练模型。')

    poem = begin_word
    word = begin_word
    with torch.no_grad():
        while word != end_token:
            input = np.array([word_int_map[w] for w in poem],dtype= np.int64)
            input = Variable(torch.from_numpy(input)).to(device)
            output = rnn_model(input, is_test=True)
            word = to_word(output.data.tolist()[-1], vocabularies)
            poem += word
            # print(word)
            # print(poem)
            if len(poem) > 50:
                break
    return poem



# run_training()  # 如果不是训练阶段 ，请注销这一行 。 网络训练时间很长。


pretty_print_poem(gen_poem("日"))
pretty_print_poem(gen_poem("红"))
pretty_print_poem(gen_poem("山"))
pretty_print_poem(gen_poem("夜"))
pretty_print_poem(gen_poem("湖"))
pretty_print_poem(gen_poem("海"))
pretty_print_poem(gen_poem("月"))

