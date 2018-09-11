"""
The abstract class to define the structure of the models. other GAN models have
to be conformed to this code architecture
"""


class GAN(object):
    def __init__(self, *args, **kwargs):
        super(GAN, self).__init__()

    def generator(self, x, params=None):
        """
        generator pass
        :param x: input tensor
        :param params: parameters for other specific actions like dropout, leaky relu ...
        :return:
        """
        pass

    def discriminator(self, x, params=None):
        """
        discriminator pass
        :param x: input tensor
        :param params: parameters for other specific actions like dropout, leaky relu ...
        :param x:
        :param params:
        :return:
        """
        pass

    def loss(self, d_loss=None, g_loss=None, params=None):
        """
        Loss function, combines the generator and discriminator losses together
        :param d_loss:
        :param g_loss:
        :param params:
        :return:
        """
        pass

    def loss_d(self, d_real=None, d_fake=None, params=None):
        """
        Calculate discrimination loss.
        :param d_real:
        :param d_fake:
        :param params:
        :return:
        """
        pass

    def loss_g(self, d_fake=None, params=None):
        """
        Calculate generator loss
        :param d_fake:
        :param params:
        :return:
        """
        pass

    def training_runner(self, training_data, validation_data,
                        testing_data=None, configs=None,
                        output_folder=None):
        """
        Load data, model and configs, set up session and start training.
        :param training_data:
        :param validation_data:
        :param testing_data:
        :param configs:
        :param output_folder:
        :return:
        """
        pass

    def __initialize_models__(self, params=None):
        """
        Initialize models, setup graph
        :param params:
        :return:
        """
        pass

    def __initialize_generator__(self, params=None):
        """
        Initialize generator model
        :param params:
        :return:
        """
        pass

    def __initialize_discriminator__(self, g_z, params=None):
        """
        Initialize discriminator model
        :param params:
        :return:
        """
        pass

    def show_result(self, num_epoch,
                    sess, g_z,
                    drop_out, z,
                    show=False, save=False,
                    path='result.png', is_fix=False):
        """
        show generator results
        :param sess: tensorflow session
        :param g_z: initialized generator model object
        :param show:
        :param save:
        :param path:
        :param is_fix:
        :return:
        """