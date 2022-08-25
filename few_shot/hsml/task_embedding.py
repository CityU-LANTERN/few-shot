import torch
import torch.nn as nn
import torch.nn.functional as F
from few_shot.hsml.image_embedding import ImageEmbedding


class TaskEmbedding(nn.Module):
    """
    Embed task to get task representation using support set.
    Output autoencoder reconstruction loss with task representation.

    Attributes:
        autoencoder: antoencoder network for task embedding.
    """
    def __init__(self, args, task_emb_dim):
        """
        Choose different autoencoder for task embedding.
        Args:
            :param task_emb_dim: typically 128
        """
        super(TaskEmbedding, self).__init__()
        self.args = args
        self.is_regression = True if hasattr(self.args, "is_regression") and self.args.is_regression else False
        if not self.is_regression:
            self.imageEmbedding = ImageEmbedding(args)
        self.ae_type = args.ae_type
        if self.ae_type == 'mean':
            self.autoencoder = MeanAutoencoder(args, task_emb_dim)
        elif self.ae_type == 'gru':
            self.autoencoder = LSTMAutoencoder(args, task_emb_dim)
        else:
            raise Exception('Autoencoder type unrecognized! choose \'mean\' or \'rnn\'. ')

    def forward(self, xs, ys, medium_record=False):
        """
        Args:
            :param xs: support set with images tensor, [batch_size,3,84,84] [NK,3,84,84]
            :param ys: support set with labels [batch_size,] [NK,]
            :param medium_record: if record medium_batch
        Returns:
            :return task_emb_vec: task embed vector [1,128]
            :return task_emb: task embed vector [NK,128]
            :return loss_rec: reconstruction loss
            :return img_emb: imageEmbedding results [NK, 64]
            :return task_emb_input: img_emb cat with y_one_hot
        """
        # medium_batch
        medium_batch = {'img_emb': [], 'task_emb_vec': []}

        if not self.is_regression:
            img_emb = self.imageEmbedding(xs)    # out shape (N*Ks, 64)
            one_hot_ys = F.one_hot(ys)    # N*Ks,eg:[10] from 0->4,2-shot 5-way
            # one_hot_ys: size: [N*Ks, N]
            task_emb_input = torch.cat((img_emb, one_hot_ys), dim=-1)    # task_emb shape(N*Ks, 64+N), dim: -1 last dim

            if medium_record:
                medium_batch['img_emb'] = img_emb.cpu().detach().numpy()

        else:  # regression
            task_emb_input = torch.cat((xs, ys.unsqueeze(1)), dim=-1)     # [N*K, 2+1]

        task_emb_vec, task_emb, loss_rec = self.autoencoder(task_emb_input)

        if medium_record:
            medium_batch['task_emb_vec'] = task_emb_vec.cpu().detach().numpy()

        return task_emb_vec, task_emb, loss_rec, medium_batch


class MeanAutoencoder(nn.Module):
    """
    Mean pooling autoencoder aggregator for task embedding.
    Input: 1 task: [NK, 64+N]: [num_imgs, img_embed] for imgs in support set.
    Output: task_emb_vec: [1, task_emb_dim], loss_rec: reconstruction loss
    # TBD! inputs -> [num_task, NK, 64+N], output -> [num_task, 128]

    Attributes:
        task_emb_dim: task representation dim. typically, 128 for img classification.
        hidden_num: number of hidden nodes in the mid layer.
        elem_num: dim for 1 input sample, [64+N], N-way
        loss_fn: MSE, task-wise
    """
    def __init__(self, args, task_emb_dim):
        """
        Args:
            :param task_emb_dim: typically 128.
        """
        super(MeanAutoencoder, self).__init__()
        self.args = args
        self.task_emb_dim = task_emb_dim
        self.hidden_num = 96
        self.elem_num = 64+args.num_classes_per_set
        self.loss_fn = nn.MSELoss()
        self.encoder = nn.Sequential(
            nn.Linear(self.elem_num, self.hidden_num),
            nn.ReLU(),
            nn.Linear(self.hidden_num, self.task_emb_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.task_emb_dim, self.hidden_num),
            nn.ReLU(),
            nn.Linear(self.hidden_num, self.elem_num),
        )
        self.last_fc = nn.Sequential(
            nn.Linear(self.task_emb_dim, self.task_emb_dim),
            nn.ReLU(),
        )

    def forward(self, inputs):
        """
        Pass a task to get task_emb_vec and reconstruction loss.

        Args:
            :param inputs: [NK, 64+N] NK image_embs, each has len 64+N

        Returns:
            :return task_emb_vec: task embed vector [1,128]
            :return loss_rec: reconstruction loss
        """
        task_emb = self.encoder(inputs)         # [NK, 128]
        inputs_rec = self.decoder(task_emb)     # [NK, 64+N]
        # loss_rec = 0.5 * torch.mean(torch.square(inputs - inputs_rec))
        loss_rec = 0.5 * self.loss_fn(inputs, inputs_rec)   # loss_rec: MSE, R
        task_emb_vecs = self.last_fc(task_emb)  # [NK, 128]
        task_emb_vec = torch.mean(task_emb_vecs, dim=0, keepdim=True)   # [1, 128]

        return task_emb_vec, task_emb_vecs, loss_rec


class LSTMAutoencoder(nn.Module):
    """
    GRU autoencoder.
    Input: 1 task: [NK, 64+N]: [num_imgs, img_embed] for imgs in support set.
    Output: task_emb_vec: [1, task_emb_dim], loss_rec: reconstruction loss

    Attributes:
        task_emb_dim: task representation dim. typically, 128 for img classification.
        elem_num: dim for 1 input sample, [64+N], N-way
        loss_fn: MSE, task-wise
    """
    def __init__(self, args, task_emb_dim):
        """
        Args:
            :param task_emb_dim: typically 128.
        """
        super(LSTMAutoencoder, self).__init__()
        self.args = args
        self.task_emb_dim = task_emb_dim
        if hasattr(self.args, "is_regression") and self.args.is_regression:
            self.elem_num = 1 + 1
        else:
            self.elem_num = 64 + args.num_classes_per_set
        self.loss_fn = nn.MSELoss()
        self.encoder = nn.GRU(input_size=self.elem_num, hidden_size=self.task_emb_dim)
        self.decoder = GRUdecoder(args, input_size=self.elem_num, hidden_size=self.task_emb_dim)
        self.apply(self.weight_init)    # customized initialization

    def forward(self, inputs):
        """
        Pass a task to get task_emb_vec and reconstruction loss.

        Args:
            :param inputs: [NK, 64+N] NK image_embs, each has len 64+N

        Returns:
            :return task_emb_vec: task embed vector [1,128]
            :return loss_rec: reconstruction loss
        """
        enc_inputs = torch.unsqueeze(inputs, dim=1)     # [NK, 1, 64+N]
        task_emb, enc_state = self.encoder(enc_inputs)
        # task_emb: [NK,1,128], enc_state: (num_layers, batch, hidden_size), [1,1,128]
        dec_outputs = self.decoder(torch.zeros(enc_inputs[0].shape).float().to(inputs.device), enc_state[0],
                                   seq_len=inputs.shape[0])     # [1,NK,N+64]
        dec_outputs = dec_outputs.squeeze(0)    # [NK, N+64]
        loss_rec = self.loss_fn(inputs, dec_outputs)   # loss_rec: R
        task_emb_vec = torch.mean(task_emb, dim=0)   # [1, 128]

        return task_emb_vec, task_emb.view(task_emb.shape[0], -1), loss_rec

    @staticmethod
    def weight_init(m):

        if isinstance(m, nn.Linear):    # for fc
            truncated_normal_(m.weight, mean=0, std=1.0)    # tf.truncated_normal((shape), stddev=1.0)
            nn.init.constant_(m.bias, 0.1)        # tf.constant(0.1, shape=[self.elem_num])
        if isinstance(m, nn.GRUCell):   # for GRUCell
            nn.init.xavier_uniform_(m.weight_ih)
            nn.init.xavier_uniform_(m.weight_hh)
            nn.init.zeros_(m.bias_ih)
            tmp = torch.ones(m.bias_hh.shape)
            tmp[int(tmp.shape[0]/3*2):] = 0.0   # tf, gates_bias=1.0, candidate_bias=0.0
            m.bias_hh = torch.nn.Parameter(tmp)
        if isinstance(m, nn.GRU):   # for GRU
            nn.init.xavier_uniform_(m.weight_ih_l0)
            nn.init.xavier_uniform_(m.weight_hh_l0)
            nn.init.zeros_(m.bias_ih_l0)
            tmp = torch.ones(m.bias_hh_l0.shape)
            tmp[int(tmp.shape[0]/3*2):] = 0.0   # tf, gates_bias=1.0, candidate_bias=0.0
            m.bias_hh_l0 = torch.nn.Parameter(tmp)


class GRUdecoder(nn.Module):
    """
    GRU decoder.
    the output of each iter -> fc (hidden_size -> input_size) -> input of the next iter.

    Attributes:
        grucell: GRUCell
        fc: hidden_size -> input_size
    """
    def __init__(self, args, input_size, hidden_size):
        super(GRUdecoder, self).__init__()
        self.args = args
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.grucell = nn.GRUCell(input_size=input_size, hidden_size=hidden_size)
        self.fc = nn.Linear(hidden_size, input_size)  # used to transform from output to input

    def forward(self, inputs, h_0, seq_len):
        """
        Args:
            :param inputs: typically torch.zeros as init input  [batch, input_size], [1, 64+N]
            :param h_0: hidden state, typically from encoder [batch, hidden_size], [1, 128]
            :param seq_len: number of images, NK
        Returns:
            :return dec_outputs: reconstruction [1,NK,N+64]
        """
        dec_outputs = []    # NK * [1,64]
        for step in range(seq_len):
            h_0 = self.grucell(inputs, h_0)
            inputs = self.fc(h_0)
            dec_outputs.append(inputs)

        dec_outputs = dec_outputs[::-1]     # reverse
        dec_outputs = torch.stack(dec_outputs, dim=1)
        return dec_outputs


def truncated_normal_(tensor, mean=0, std=0.09):
    with torch.no_grad():
        size = tensor.shape
        tmp = tensor.new_empty(size+(4,)).normal_()     # mean 0, std 1
        valid = (tmp < 2) & (tmp > -2)
        ind = valid.max(-1, keepdim=True)[1]
        tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
        tensor.data.mul_(std).add_(mean)

    return tensor


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser("")
    # ImageEmbedding
    parser.add_argument('--image-height', default=84, type=int)
    parser.add_argument('--image-width', default=84, type=int)
    # HierarchicalClustering
    parser.add_argument('--cluster-layer-0', default=4, type=int)
    parser.add_argument('--cluster-layer-1', default=4, type=int)
    parser.add_argument('--sigma', default=10.0, type=float)
    # TaskEmbedding
    parser.add_argument('--is-regression', default=False, type=bool)
    parser.add_argument('--ae-type', default='gru', type=str)
    parser.add_argument('--num-classes-per-set', default=5, type=int)
    args = parser.parse_args()

    model = TaskEmbedding(args, task_emb_dim=128)
    # TaskEmbedding(
    #   (imageEmbedding): ImageEmbedding(
    #     (conv): Sequential(
    #       (0): Conv2d(3, 32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    #       (1): ReLU()
    #       (2): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
    #       (3): LocalResponseNorm(4, alpha=0.00011111111111111112, beta=0.75, k=1.0)
    #       (4): Conv2d(32, 32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    #       (5): ReLU()
    #       (6): LocalResponseNorm(4, alpha=0.00011111111111111112, beta=0.75, k=1.0)
    #       (7): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
    #       (8): Flatten(start_dim=1, end_dim=-1)
    #     )
    #     (fc): Sequential(
    #       (0): Linear(in_features=14112, out_features=384, bias=True)
    #       (1): LeakyReLU(negative_slope=0.01)
    #       (2): Linear(in_features=384, out_features=64, bias=True)
    #       (3): LeakyReLU(negative_slope=0.01)
    #     )
    #   )
    #   (autoencoder): LSTMAutoencoder(
    #     (loss_fn): MSELoss()
    #     (encoder): GRU(69, 128)
    #     (decoder): GRUdecoder(
    #       (grucell): GRUCell(69, 128)
    #       (fc): Linear(in_features=128, out_features=69, bias=True)
    #     )
    #   )
    # )
    x = torch.rand(5,3,84,84)       # with all 5 labels
    y = torch.tensor([0,1,2,3,4])
    pred = model(x, y, medium_record=True)
    # pred {Tensor: {1, 128}, Tensor: {5, 128}, Tensor: {},
    #       {'img_emb': ndarray {5, 64}, 'task_emb_vec': ndarray {1, 128}}}
