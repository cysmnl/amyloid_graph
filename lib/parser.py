import argparse, sys

parser = argparse.ArgumentParser()

parser.add_argument('-m3', '--model3', help='specify if model accepts a 3D tensor as input',
    action="store_true")

parser.add_argument('-tr', '--train', help='specify if training',
    action="store_true")

parser.add_argument('-e', '--eval', help='evalute the model via 10 fold validation',
    action="store_true")

parser.add_argument(
    '--dir_name',
    default='test',
    type=str)

parser.add_argument(
    '--epochs',
    default=100)

parser.add_argument(
    '--batch',
    default=20)

parser.add_argument(
    '--filters',
    default='[8,16]',
    type=str)

parser.add_argument(
    '--poly_order',
    default='[20,20]',
    type=str)

parser.add_argument(
    '--fc',
    default='[128,2]',
    type=str)

parser.add_argument(
    '--reg',
    default=5e-4)

parser.add_argument(
    '--dropout',
    default='1')

parser.add_argument(
    '--learn_rate',
    default=1e-3)

parser.add_argument(
    '--decay_rate',
    default=1)

parser.add_argument(
    '--momentum',
    default=0)

parser.add_argument(
    '--covar',
    default='null')

parser.add_argument(
    '--trainx')

parser.add_argument(
    '--trainy')

parser.add_argument(
    '--testx')

parser.add_argument(
    '--testy')