INSTRUCTIONS = {
    "reach-v1": ["reach to goal_pos", 'reach goal_pos'],
    'push-v1': ["push goal_pos", 'push to goal_pos', 'push object to goal_pos', 'push block to goal_pos'],
    'pick-place-v1': ["pick and place at goal_pos", 'pick object and place at goal_pos'],
    'door-open-v1': ["pull goal_pos", 'open door', 'pull to goal_pos'],
    'door-close-v1': ['push to goal_pos', 'close door', 'push from left'],
    'drawer-open-v1': ["pull goal_pos", 'pull to goal_pos', 'pull back to goal_pos'],
    'drawer-close-v1': ["push goal_pos", 'push to goal_pos', 'push forward to goal_pos'],
    'button-press-topdown-v1': ["push object down to goal_pos", 'press button', 'press down', 'press button down'],
    'peg-insert-side-v1': [""],
    'window-open-v1': ['push right to goal_pos', 'push object right', 'slide object left', 'open window'],
    'window-close-v1': ['push left to goal_pos', 'push object left', 'slide object right', 'close window'],
    'reach-wall-v1': ['reach over obstacle to goal_pos', 'reach goal_pos', 'reach to goal_pos'],
    'pick-place-wall-v1': ['pick object and place at goal_pos', 'pick and place at goal_pos'],
    'push-wall-v1': ['push object around obstacle to goal_pos', 'push to goal_pos', 'push object to goal_pos'],
    'button-press-v1': ['press forward', 'press button forward', 'push to goal_pos'],
    'button-press-topdown-wall-v1': ['press behind obstacle', 'press down to goal_pos', 'press down'],
    'button-press-wall-v1': ['press button behind obstacle', 'press forward', 'press forward to goal_pos'],
    'peg-unplug-side-v1': ['pull object to right', 'pull object to goal_pos', 'pick object and pull to right'],
    'disassemble-v1': ['pick and pull up', 'pick and place at goal_pos', 'pick and put down at goal_pos'],
    'hammer-v1': ['push to goal_pos with object', 'push to goal_pos with hammer', 'use object to push to goal_pos'],
    'plate-slide-v1': ['push to goal_pos', 'push plate to goal_pos', 'push forward to goal_pos'],
    'plate-slide-side-v1': ['push left to goal_pos', 'push plate left to goal_pos', 'slide left to goal_pos'],
    'plate-slide-back-v1': ['push back to goal_pos', 'push plate back to goal_pos', 'slide back to goal_pos'],
    'plate-slide-back-side-v1': ['push right to goal_pos', 'push plate right to goal_pos', 'slide right to goal_pos'],
    'handle-press-v1': ['push down', 'press down', 'push down to goal_pos', 'press down to goal_pos'],
    'handle-pull-v1': ['pull up', 'push up', 'push up to goal_pos', 'pull up to goal_pos'],
    'handle-press-side-v1': ['push down', 'press down', 'push down to goal_pos', 'press down to goal_pos'],
    'handle-pull-side-v1': ['pull up', 'push up', 'push up to goal_pos', 'pull up to goal_pos'],
    'stick-push-v1': ['push with object to goal_pos', 'use stick to push to goal_pos', 'push with stick to goal_pos'],
    'stick-pull-v1': ['push with object to goal_pos', 'use stick to push to goal_pos', 'push with stick to goal_pos'],
    'basketball-v1': ['pick and place at goal_pos', 'pick ball and place at goal_pos', 'pick ball and place at goal_pos from the top'],
    'soccer-v1': ['push to goal_pos', 'push ball to goal_pos', 'place object at goal_pos', 'place ball at goal_pos'],
    'faucet-open-v1': ['open faucet', 'push to goal_pos', 'push right', 'push from left to right',
                       'turn object from left to right', 'turn to goal_pos'],
    'faucet-close-v1': ['close faucet', 'push to goal_pos', 'push left', 'push from right to left',
                        'turn object from right to left', 'turn to goal_pos'],
    'coffee-push-v1': ['push object to goal_pos', 'push cup to goal_pos', 'push object forward to goal_pos'],
    'coffee-pull-v1': ['place object away', 'pick and place at goal_pos', 'place cup away'],
    'coffee-button-v1': ['push forward', 'press button', 'press coffee button'],
    'sweep-v1': ['sweep', 'sweep object from table', 'sweep block from table', 'slide object from table', 'slide block from table'],
    'sweep-into-v1': ['sweep into hole', 'push to goal_pos', 'push block to goal_pos', 'push block into hole'],
    'pick-out-of-hole-v1': ['pick object', 'pick object out of hole', 'pick and place at goal_pos'],
    'assembly-v1': ['pick and place at goal_pos', 'assemble', 'pick object and place down at goal_pos'],
    'shelf-place-v1': ['pick and place at goal_pos', 'place on shelf at goal_pos', 'pick object and place on shelf at goal_pos'],
    'push-back-v1': ['push back to goal_pos', 'push to goal_pos', 'puch block to goal_pos',
                     'push block back to goal_pos'],
    'lever-pull-v1': ['pull to goal_pos'],
    'dial-turn-v1': ['turn object', 'dial turn', 'rotate object'],

    'bin-picking-v1': ['pick object', 'pick and place at goal_pos'],
    'box-close-v1': ['pick object and place at goal_pos', 'pick and place at goal_pos'],
    'hand-insert-v1': ['sweep object to goal_pos', 'pick object and place at goal_pos', 'push object to goal_pos',
                       'push block to goal_pos'],
    'door-lock-v1': ['push object down to goal_pos', 'turn object down to goal_pos'],
    'door-unlock-v1': ['push object to goal_pos', 'turn object to goal_pos'],
}

from flair.embeddings import WordEmbeddings, DocumentPoolEmbeddings, TransformerDocumentEmbeddings
from flair.data import Sentence


word_embedding = WordEmbeddings('glove')
document_embedding = DocumentPoolEmbeddings([word_embedding])
document_embedding_roberta = TransformerDocumentEmbeddings('roberta-base')

embeddings = dict()
for key, val in INSTRUCTIONS.items():
    k = key.replace('-v1', '')
    k = k.replace('-', ' ')
    print(k)
    sentence = Sentence(k)
    document_embedding_roberta.embed(sentence)

    embeddings[k] = sentence.embedding.cpu().numpy()
    print(sentence.embedding.shape)

with open('contenxt_embeddings_roberta.pkl', 'wb') as handle:
    import pickle

    pickle.dump(embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL)


