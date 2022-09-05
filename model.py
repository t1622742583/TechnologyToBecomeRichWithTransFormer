import math
import torch
from torch import nn
from torch.nn.functional import softmax


def t2v(tau, f, out_features, w, b, w0, b0, arg=None):
    if arg:
        v1 = f(torch.matmul(tau, w) + b, arg)
    else:
        # print(w.shape, t1.shape, b.shape)
        v1 = f(torch.matmul(tau, w) + b)
    v2 = torch.matmul(tau, w0) + b0
    # print(v1.shape)
    return torch.cat([v1, v2], -1)


class SineActivation(nn.Module):
    def __init__(self, in_features, ba_size, out_features):
        # i = 6 out = 5
        super(SineActivation, self).__init__()
        self.out_features = out_features
        # 6x1
        self.w0 = nn.parameter.Parameter(torch.randn(ba_size, in_features, 1))
        # 1
        self.b0 = nn.parameter.Parameter(torch.randn(ba_size, 32, 1))
        # 6x4
        self.w = nn.parameter.Parameter(torch.randn(ba_size, in_features, out_features - 1))
        # 4
        self.b = nn.parameter.Parameter(torch.randn(ba_size, 32, out_features - 1))
        self.f = torch.sin

    def forward(self, tau):
        return t2v(tau, self.f, self.out_features, self.w, self.b, self.w0, self.b0)


class CosineActivation(nn.Module):
    def __init__(self, in_features, ba_size, out_features):
        super(CosineActivation, self).__init__()
        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.randn(ba_size, in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(ba_size, 1))
        self.w = nn.parameter.Parameter(torch.randn(ba_size, in_features, out_features - 1))
        self.b = nn.parameter.Parameter(torch.randn(ba_size, out_features - 1))
        self.f = torch.cos

    def forward(self, tau):
        return t2v(tau, self.f, self.out_features, self.w, self.b, self.w0, self.b0)


class Time2Vec(nn.Module):
    def __init__(self, activation, hidden_dim):
        '''
        :param activation: 激活函数（非线性激活函数） sin/cos
        :param hidden_dim: 隐藏（自定义，不影响运行）
        '''
        super(Time2Vec, self).__init__()
        if activation == 'sin':
            self.activation = torch.sin
        else:
            self.activation = torch.cos
        self.out_features = hidden_dim
        self.fc1 = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        device = x.device
        # 获取x的尺寸信息
        batch_size = x.shape[0]
        sentence_len = x.shape[1]
        in_features = x.shape[2]
        # 初始化权重和偏置
        self.w0 = nn.parameter.Parameter(torch.randn(batch_size, in_features, 1)).to(device)
        self.b0 = nn.parameter.Parameter(torch.randn(batch_size, sentence_len, 1)).to(device)
        self.w = nn.parameter.Parameter(torch.randn(batch_size, in_features, self.out_features - 1)).to(device)
        self.b = nn.parameter.Parameter(torch.randn(batch_size, sentence_len, self.out_features - 1)).to(device)
        # 运算
        v1 = self.activation(torch.matmul(x, self.w) + self.b)
        v2 = torch.matmul(x, self.w0) + self.b0
        v3 = torch.cat([v1, v2], -1)
        x = self.fc1(v3)
        return x


class LSTM_Attention(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, n_layers, num_class):
        super(LSTM_Attention, self).__init__()
        self.model_type = 'bi_lstm_attention'  # 模型
        # 从LSTM得到output之后，将output通过下面的linear层，然后就得到了Q,K,V
        # 这里我是用的attention_size是等于hidden_dim的，这里可以自己换成别的attention_size
        self.W_Q = nn.Linear(hidden_dim * 2, hidden_dim * 2, bias=False)
        self.W_K = nn.Linear(hidden_dim * 2, hidden_dim * 2, bias=False)
        self.W_V = nn.Linear(hidden_dim * 2, hidden_dim * 2, bias=False)

        # embedding层
        # self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # LSTM
        self.rnn = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=n_layers, bidirectional=True,
                           dropout=0.3)
        # Linear层,因为是三分类，所以后面的维度为3
        self.fc = nn.Linear(hidden_dim * 2, num_class)
        # dropout
        self.dropout = nn.Dropout(0.3)

    # 用来计算attention
    def attention(self, Q, K, V):
        d_k = K.size(-1)
        scores = torch.matmul(Q, K.transpose(1, 2)) / math.sqrt(d_k)
        alpha_n = softmax(scores, dim=-1)
        context = torch.matmul(alpha_n, V)

        # 这里都是组合之后的矩阵之间的计算，所以.sum之后，得到的output维度就是[batch_size,hidden_dim]，并且每一行向量就表示一句话，所以总共会有batch_size行
        output = context.sum(1)

        return output, alpha_n

    def forward(self, x):
        # x.shape = [seq_len,batch_size] = [30,64]

        # embedding = self.dropout(self.embedding(x))  # embedding.shape = [seq_len,batch_size,embedding_dim = 100]
        # embedding = embedding.transpose(0, 1)  # embedding.shape = [batch_size,seq_len,embedding_dim]
        # 进行LSTM
        output, (h_n, c) = self.rnn(x)  # out.shape = [batch_size,seq_len,hidden_dim=128]

        Q = self.W_Q(output)  # [batch_size,seq_len,hidden_dim]
        K = self.W_K(output)
        V = self.W_V(output)

        # 将得到的Q，K，V送入attention函数进行运算
        attn_output, alpha_n = self.attention(Q, K, V)
        # attn_output.shape = [batch_size,hidden_dim=128]
        # alpha_n.shape = [batch_size,seq_len,seq_len]

        out = self.fc(attn_output)  # out.shape = [batch_size,num_class]
        return out


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # max_len 防止词向量太大
        # 生成5000x11的0矩阵
        pe = torch.zeros(max_len, d_model)
        # 生成0-4999w位置 信息 结构 5000x1
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # torch.exp e的x次方 以e为底的指数函数
        # torch.arange(0, d_model, 2) 0 到 11的数间隔2  0,2,4,8,10
        bm = (-math.log(10000.0) / d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransAm(nn.Module):
    def __init__(self, s_len=30, feature_size=5, num_layers=1, dropout=0.1, nhead=11, num_class=2, device='cpu'):
        super(TransAm, self).__init__()
        feature_size += 2
        nhead += 2
        self.s_len = s_len
        self.model_type = 'Transformer'
        self.device = device
        self.time2Voc = Time2Vec('sin', 5, self.device).to(self.device)
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(feature_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(feature_size * s_len, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        time_src = self.time2Voc(src)
        src = torch.cat([src, time_src], dim=-1)
        # if self.src_mask is None or self.src_mask.size(0) != len(src):
        #     device = src.device
        #     mask =
        #     self.src_mask = mask
        self.src_mask = self._generate_square_subsequent_mask(len(src)).to(self.device)
        # src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = output.view(src.size()[0], -1)
        output = self.decoder(output)
        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz,sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
class TransfomerWithTimeEncoder(nn.Module):
    def __init__(self, s_len, feature_size, num_layers, dropout):
        '''

        :param s_len: 窗口长度
        :param feature_size: 特征数量
        :param num_layers: tranformer叠加层数
        :param dropout:
        :param n_head: 多头(与特征数量整除)
        '''
        super().__init__()
        feature_size += 2
        n_head = feature_size
        self.s_len = s_len
        self.model_type = 'Transformer'
        self.time2Voc = Time2Vec('sin', 5)
        self.src_mask = None
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=n_head, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz,sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    def forward(self, src):
        device = src.device
        time_src = self.time2Voc(src)
        src = torch.cat([src, time_src], dim=-1)
        self.src_mask = self._generate_square_subsequent_mask(len(src)).to(device)
        output = self.transformer_encoder(src, self.src_mask)
        output = output.view(src.size()[0], -1)
        return output
class ModestCattle(nn.Module):
    '''谦牛模型'''
    def __init__(self, num_layers, dropout, n_class,k_len,k_feature_size,trick_len,trick_feature_size):
#       k线编码器
        super().__init__()
        self.model_type = 'modest_cattle'
        s_len = 64
        # feature_size,n_head = 11
        self.k_encoder = TransfomerWithTimeEncoder(s_len=k_len, feature_size=k_feature_size, num_layers=num_layers, dropout=dropout)
#         时编码器
        s_len = 47
        # feature_size,n_head = 5
        self.trick_encoder = TransfomerWithTimeEncoder(s_len=trick_len, feature_size=trick_feature_size, num_layers=num_layers, dropout=dropout)
#         文字编码器
#         解码器
        self.conver = nn.Linear(trick_len*(trick_feature_size+2), k_len*(k_feature_size+2))
        self.decoder = nn.Linear(k_len*(k_feature_size+2), n_class)
    def forward(self, k_x,trick_x):
#         当前标的k线编码
        k_x = self.k_encoder(k_x)
#         batch_size,k_len = k_x.shape
#         当前标的分时编码
        trick_x = self.trick_encoder(trick_x)
        trick_x = self.conver(trick_x)

        # pass
#         当前大盘k线编码
#         当前分时分时编码
#         当前评论编码
#         当前咨询编码
        all_information = k_x + trick_x
        return self.decoder(all_information)
if __name__ == '__main__':
    # time2vec = Time2Vec("sin", 5)
    k_x = torch.randn((1, 64, 9)).to('cuda')
    trick_x = torch.randn((1, 47, 6)).to('cuda')
    # m = time2vec(n)
    trans = ModestCattle(num_layers=1, dropout=0.3,n_class=2,k_len=64,k_feature_size=9,trick_len=47,trick_feature_size=6).to('cuda')
    m = trans(k_x,trick_x)
    pass
