import torch
import torch.nn as nn


class ImageEmbedding(nn.Module):
    """
    Embed images to get in a CNN to get image representation for support x.
    args needs to contain image_height and image_width
    Input dim: image shape;
    Output dim: [64,1].

    structure:
        input -> conv1 -> relu -> max pool -> local response norm ->conv2 -> relu -> local response norm -> max pool ->
        flatten -> dense -> relu -> dense -> relu -> output.

    Attributes:
        conv: 2 conv layers, output is the flattened features.
        fc: 2 dense layers, output size is 64.
    """
    def __init__(self, args):
        super(ImageEmbedding, self).__init__()
        self.args = args
        hidden_num = 32  # num of nodes in hidden layer for 2 convs, typically 32: 3->hidden_num;hidden_num->hidden_num.
        channels = 3     # number of channels, 3 for RGB images, 1 for regression.
        image_height = args.image_height    # size of images, typically 84
        image_width = args.image_width     # size of images, typically 84
        self.conv = nn.Sequential(
            nn.Conv2d(channels, hidden_num, kernel_size=5, stride=1, padding=2),    # same padding
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),   # same padding
            nn.LocalResponseNorm(size=4, alpha=0.001/9.0, beta=0.75, k=1.0),
            nn.Conv2d(hidden_num, hidden_num, kernel_size=5, stride=1, padding=2),  # same padding
            nn.ReLU(),
            nn.LocalResponseNorm(size=4, alpha=0.001/9.0, beta=0.75, k=1.0),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),   # same padding
            nn.Flatten(),
        )

        with torch.no_grad():
            sample_inp = torch.rand(2, channels, image_height, image_width)
            sample_out = self.conv(sample_inp)
            input_size = sample_out.shape[1]    # typically, (84/2/2)*(84/2/2)*hidden_num=14112

        self.fc = nn.Sequential(
            nn.Linear(input_size, 384),
            nn.LeakyReLU(),
            nn.Linear(384, 64),
            nn.LeakyReLU(),
        )
        self.apply(self.weight_init)    # customized initialization

    def forward(self, x):
        """
        Args:
            :param x: images tensor with size [batch_size,3,84,84]
        Returns:
            :return out: image embedding with size [batch_size,64]
        """
        out = self.conv(x)  # out shape(batch,hidden_num*21*21)
        # out = out.view(out.size(0), -1)
        out = self.fc(out)      # out shape(batch,64)
        return out

    @staticmethod
    def weight_init(m):

        if isinstance(m, nn.Linear):    # for fc
            truncated_normal_(m.weight, mean=0, std=0.04)   # tf.truncated_normal_initializer(stddev=0.04)
            nn.init.constant_(m.bias, 0.1)                  # tf.constant_initializer(0.1)

        elif isinstance(m, nn.Conv2d):  # for conv
            truncated_normal_(m.weight, mean=0, std=0.04)   # tf.truncated_normal_initializer(stddev=0.04)


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
    args = parser.parse_args()

    model = ImageEmbedding(args)
    # ImageEmbedding(
    #   (conv): Sequential(
    #     (0): Conv2d(3, 32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    #     (1): ReLU()
    #     (2): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
    #     (3): LocalResponseNorm(4, alpha=0.00011111111111111112, beta=0.75, k=1.0)
    #     (4): Conv2d(32, 32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    #     (5): ReLU()
    #     (6): LocalResponseNorm(4, alpha=0.00011111111111111112, beta=0.75, k=1.0)
    #     (7): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
    #     (8): Flatten(start_dim=1, end_dim=-1)
    #   )
    #   (fc): Sequential(
    #     (0): Linear(in_features=14112, out_features=384, bias=True)
    #     (1): LeakyReLU(negative_slope=0.01)
    #     (2): Linear(in_features=384, out_features=64, bias=True)
    #     (3): LeakyReLU(negative_slope=0.01)
    #   )
    # )
    x = torch.rand(1,3,84,84)
    pred = model(x)         # pred {Tensor: {1, 64}}
