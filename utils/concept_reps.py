from matplotlib import pyplot as plt

plt.style.use('default')
from utils.analysis_from_interaction import *
import os

if not os.path.exists('analysis'):
    os.makedirs('analysis')
from collections import Counter


def objects_to_concepts(sender_input, n_values):
    """reconstruct concepts from objects in interaction"""
    n_targets = int(sender_input.shape[1] / 2)
    # get target objects and fixed vectors to re-construct concepts
    target_objects = sender_input[:, :n_targets]
    target_objects = k_hot_to_attributes(target_objects, n_values)
    # concepts are defined by a list of target objects (here one sampled target object) and a fixed vector
    (objects, fixed) = retrieve_concepts_sampling(target_objects, all_targets=True)
    return list(zip(objects, fixed))


def objects_to_context(sender_input, n_values):
    """reconstruct context conditions from objects in interaction"""
    n_targets = int(sender_input.shape[1] / 2)
    # get targets and distractors to reconstruct concept and context conditions
    targets = sender_input[:, :n_targets]
    targets = k_hot_to_attributes(targets, n_values)
    distractors = sender_input[:, n_targets:]
    distractors = k_hot_to_attributes(distractors, n_values)
    (objects, fixed) = retrieve_concepts_sampling(targets, all_targets=True)
    return retrieve_context_condition(targets, fixed, distractors)


def retrieve_messages(interaction, is_gumbel=True):
    """retrieve messages from interaction"""
    if is_gumbel:
        messages = interaction.message.argmax(dim=-1)
        messages = [msg.tolist() for msg in messages]
    else:
        messages = interaction.message.argmax(dim=2)
        messages = [msg.tolist() for msg in messages]
    return messages


def remove_symbs_after_eos(message):
    """
    Trims a message to the first EOS symbol, i.e. 0.
    """
    try:
        return message[:message.index(0)]
    except ValueError:
        return message


def count_symbols(messages):
    """counts symbols in messages"""
    all_symbols = [symbol for message in messages for symbol in message]
    symbol_counts = Counter(all_symbols)
    return symbol_counts


def get_unique_message_set(messages):
    """returns unique messages as a set ready for set operations"""
    return set(tuple(message) for message in messages)


def get_unique_concept_set(concepts):
    """returns unique concepts"""
    concept_tuples = []
    for objects, fixed in concepts:
        tuple_objects = []
        for object in objects:
            tuple_objects.append(tuple(object))
        tuple_objects = tuple(tuple_objects)
        tuple_concept = (tuple_objects, tuple(fixed))
        concept_tuples.append(tuple_concept)
    tuple(concept_tuples)
    unique_concepts = set(concept_tuples)
    return unique_concepts
