# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import numpy as np
import tensorflow as tf

import tfutil

#----------------------------------------------------------------------------
# Convenience func that casts all of its arguments to tf.float32.

def fp32(*values):
    if len(values) == 1 and isinstance(values[0], tuple):
        values = values[0]
    values = tuple(tf.cast(v, tf.float32) for v in values)
    return values if len(values) >= 2 else values[0]

#----------------------------------------------------------------------------
# Generator loss function used in the paper (WGAN + AC-GAN).

def G_wgan_acgan(G, D, opt, training_set, minibatch_size, unlabeled_reals,
    cond_weight = 0.0): # Weight of the conditioning term.
    '''
    Calculating the feature matching loss for the generator
    '''
    # get generated samples
    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    # get random labels for the generated samples
    rand_gen_labels = training_set.get_random_labels_tf(minibatch_size)
    # use the generator to deconvolve the latents into images
    fake_images_out = G.get_output_for(latents, rand_gen_labels, is_training=True)

    # use the discriminator to get the features from the last convolution layer as well as the logits
    fake_logits_out, _, fake_features_out = fp32(D.get_output_for(fake_images_out, is_training=False))
    # Pass the unlabeled real data to the discriminator and grab the real features out from the last convolutional layer
    _, _, real_features_out = fp32(D.get_output_for(unlabeled_reals, is_training=False))

    # calculate feature-matching loss
    # mean squared error of fake and real features
    feat_diff = tf.math.reduce_mean(fake_features_out, axis=0) - tf.math.reduce_mean(real_features_out, axis=0)
    loss = tf.math.reduce_mean(tf.math.square(feat_diff))

    loss = tfutil.autosummary('Loss/G_feat_match_loss', loss)

    # if D.output_shapes[1][1] > 0:
    #     with tf.name_scope('LabelPenalty'):
    #         # pass fake logits and labels to a softmax layer
    #         label_penalty_fakes = tf.nn.softmax_cross_entropy_with_logits_v2(labels=rand_gen_labels, logits=fake_logits_out)
    #     loss += label_penalty_fakes * cond_weight
    # loss = tfutil.autosummary('Loss/G_feat_match_loss_post_LabelPenalty', loss)
    return loss


def G_pggan_loss(G, D, opt, training_set, minibatch_size,
    cond_weight = 1.0): # Weight of the conditioning term.

    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    labels = training_set.get_random_labels_tf(minibatch_size)
    fake_images_out = G.get_output_for(latents, labels, is_training=True)
    fake_labels_out, fake_scores_out, _ = fp32(D.get_output_for(fake_images_out, is_training=True))
    loss = -fake_scores_out

    return loss

#----------------------------------------------------------------------------
# Discriminator loss function used in the paper (WGAN-GP + AC-GAN).

def D_wgangp_acgan(G, D, opt, training_set, minibatch_size, reals, labels, unlabeled_reals,
    wgan_lambda     = 0.0,      # Weight for the gradient penalty term.
    wgan_epsilon    = 0.0,      # Weight for the epsilon term, \epsilon_{drift}.
    wgan_target     = 0.1,      # Target value for gradient magnitudes.
    cond_weight     = 0.0):     # Weight of the conditioning terms.

    # Generate latents and pass through the generator to decolvolve into fake images
    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    fake_images_out = G.get_output_for(latents, labels, is_training=True)

    # REALS
    output_before_softmax_lab, real_flogit_out, _ = fp32(D.get_output_for(reals, is_training=True))
    # UNLABELED REALS
    output_before_softmax_unl, _, _ = fp32(D.get_output_for(unlabeled_reals, is_training=True))
    # GENERATED
    output_before_softmax_fake, fake_flogit_out, _ = fp32(D.get_output_for(fake_images_out, is_training=True))

    # Direct port labeled loss from Tim Salimans et al. https://arxiv.org/pdf/1606.03498.pdf
    # no support for tensor indexing, so no work
    #simple_labels = tf.argmax(labels, axis=1)
    #z_exp_lab = tf.math.reduce_mean(tf.math.reduce_logsumexp(output_before_softmax_lab, axis=1))
    #l_lab = output_before_softmax_lab[tf.range(minibatch_size), simple_labels]
    #loss_lab = -tf.math.reduce_mean(l_lab) + tf.math.reduce_mean(z_exp_lab)

    train_err = tf.math.reduce_mean(tf.cast(tf.math.not_equal(tf.math.argmax(output_before_softmax_lab, axis=1),
                                                              tf.math.argmax(labels, axis=1)), tf.float32))
    train_err = tfutil.autosummary('Loss/D_train_err', train_err)

    # labeled sample loss is equivalent to cross entropy w/ softmax (I think?)
    loss_lab = tf.math.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=output_before_softmax_lab))

    # Another implementation of Salimans code ported to TF (NOT WORKING the tf.gather is wrong)
    #l_lab = tf.gather(output_before_softmax_lab, tf.range(minibatch_size),labels)
    #loss_lab = -tf.math.reduce_sum(l_lab) + tf.math.reduce_sum(tf.math.reduce_sum(tf.math.reduce_logsumexp(output_before_softmax_lab)))

    # Direct port of unlabeled loss and fake loss. from Tim Salimans et al. https://arxiv.org/pdf/1606.03498.pdf
    # Code reference https://github.com/openai/improved-gan/blob/master/mnist_svhn_cifar10/train_cifar_feature_matching.py#L87
    #z_exp_unl = tf.math.reduce_mean(tf.math.reduce_logsumexp(output_before_softmax_unl, axis=1))
    loss_unl = -0.5*tf.math.reduce_mean(tf.math.reduce_logsumexp(output_before_softmax_unl, axis=1)) + \
               0.5*tf.math.reduce_mean(tf.math.softplus(tf.math.reduce_logsumexp(output_before_softmax_unl, axis=1)))
    loss_fake = 0.5*tf.math.reduce_mean(tf.math.softplus(tf.math.reduce_logsumexp(output_before_softmax_fake, axis=1)))

    # Using autosummary for tensorboard
    loss_lab = tfutil.autosummary('Loss/D_loss_lab', loss_lab)
    loss_unl = tfutil.autosummary('Loss/D_loss_unl', loss_unl)
    loss_fake = tfutil.autosummary('Loss/D_loss_fake', loss_fake)

    # combine losses
    loss = loss_lab + loss_unl + loss_fake + (train_err*0)

    loss = tfutil.autosummary('Loss/D_combined_loss', loss)

    # with tf.name_scope('GradientPenalty'):
    #     mixing_factors = tf.random_uniform([minibatch_size, 1, 1, 1], 0.0, 1.0, dtype=fake_images_out.dtype)
    #     mixed_images_out = tfutil.lerp(tf.cast(reals, fake_images_out.dtype), fake_images_out, mixing_factors)
    #     mixed_scores_out, mixed_labels_out, _ = fp32(D.get_output_for(mixed_images_out, is_training=True))
    #     mixed_scores_out = tfutil.autosummary('Loss/mixed_scores', mixed_scores_out)
    #     mixed_loss = opt.apply_loss_scaling(tf.reduce_sum(mixed_scores_out))
    #     mixed_grads = opt.undo_loss_scaling(fp32(tf.gradients(mixed_loss, [mixed_images_out])[0]))
    #     mixed_norms = tf.sqrt(tf.reduce_sum(tf.square(mixed_grads), axis=[1,2,3]))
    #     mixed_norms = tfutil.autosummary('Loss/mixed_norms', mixed_norms)
    #     gradient_penalty = tf.square(mixed_norms - wgan_target)
    # loss += gradient_penalty * (wgan_lambda / (wgan_target**2))

    # with tf.name_scope('EpsilonPenalty'):
    #     epsilon_penalty = tfutil.autosummary('Loss/epsilon_penalty', tf.square(real_flogit_out))
    # loss += epsilon_penalty * wgan_epsilon

    # if D.output_shapes[1][1] > 0:
    #     with tf.name_scope('LabelPenalty'):
    #         label_penalty_reals = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=output_before_softmax_lab)
    #         label_penalty_fakes = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=output_before_softmax_fake)
    #         label_penalty_reals = tfutil.autosummary('Loss/label_penalty_reals', label_penalty_reals)
    #         label_penalty_fakes = tfutil.autosummary('Loss/label_penalty_fakes', label_penalty_fakes)
    #     loss += (label_penalty_reals + label_penalty_fakes) * cond_weight

    # loss = tfutil.autosummary('Loss/D_combined_loss_post_penalties', loss)
    return loss


def D_pggan_loss(G, D, opt, training_set, minibatch_size, unlabeled_reals,
    wgan_lambda     = 10.0,     # Weight for the gradient penalty term.
    wgan_epsilon    = 0.001,    # Weight for the epsilon term, \epsilon_{drift}.
    wgan_target     = 1.0,      # Target value for gradient magnitudes.
    cond_weight     = 1.0):     # Weight of the conditioning terms.

    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    labels = training_set.get_random_labels_tf(minibatch_size)
    fake_images_out = G.get_output_for(latents, labels, is_training=True)
    real_labels_out, real_scores_out, _ = fp32(D.get_output_for(unlabeled_reals, is_training=True))
    fake_labels_out, fake_scores_out, _ = fp32(D.get_output_for(fake_images_out, is_training=True))
    real_scores_out = tfutil.autosummary('Loss/D_pggan_real_scores', real_scores_out)
    fake_scores_out = tfutil.autosummary('Loss/D_pggan_fake_scores', fake_scores_out)
    loss = fake_scores_out - real_scores_out

    with tf.name_scope('GradientPenalty'):
        mixing_factors = tf.random_uniform([minibatch_size, 1, 1, 1], 0.0, 1.0, dtype=fake_images_out.dtype)
        mixed_images_out = tfutil.lerp(tf.cast(unlabeled_reals, fake_images_out.dtype), fake_images_out, mixing_factors)
        mixed_labels_out, mixed_scores_out, _ = fp32(D.get_output_for(mixed_images_out, is_training=True))
        mixed_scores_out = tfutil.autosummary('Loss/D_pggan_mixed_scores', mixed_scores_out)
        mixed_loss = opt.apply_loss_scaling(tf.reduce_sum(mixed_scores_out))
        mixed_grads = opt.undo_loss_scaling(fp32(tf.gradients(mixed_loss, [mixed_images_out])[0]))
        mixed_norms = tf.sqrt(tf.reduce_sum(tf.square(mixed_grads), axis=[1,2,3]))
        mixed_norms = tfutil.autosummary('Loss/D_pggan_mixed_norms', mixed_norms)
        gradient_penalty = tf.square(mixed_norms - wgan_target)
    loss += gradient_penalty * (wgan_lambda / (wgan_target**2))

    with tf.name_scope('EpsilonPenalty'):
        epsilon_penalty = tfutil.autosummary('Loss/D_pggan_epsilon_penalty', tf.square(real_scores_out))
    loss += epsilon_penalty * wgan_epsilon
    return loss

#----------------------------------------------------------------------------


