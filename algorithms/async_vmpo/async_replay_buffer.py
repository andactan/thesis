import time
import numpy as np

from rlpyt.replays.async_ import AsyncReplayBufferMixin
from rlpyt.utils.buffer import (
    buffer_from_example,
    get_leading_dims,
    torchify_buffer
)
from rlpyt.utils.collections import namedarraytuple


class AsyncReplayBuffer(AsyncReplayBufferMixin):
    """Replays sequences with starting state chosen uniformly randomly
    """
    def __init__(self, example, sampler_B, optim_T, optim_B, target_steps, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.samples = buffer_from_example(example, (optim_T, optim_B), share_memory=self.async_)
        self.sampler_B = sampler_B #! unused
        self.optim_T = optim_T # sequence length
        self.optim_B = optim_B # num of environments (batch size)
        self.target_steps = target_steps
        self.sleep_length = 0.01
        self.t = 0

        self.buffer_size = optim_B * target_steps

        field_names = [field for field in example._fields if field != "prev_rnn_state"]
        self.SamplesToBuffer = namedarraytuple('SamplesToBuffer', field_names)
        
        buffer_example = self.SamplesToBuffer(*(v for k, v in example.items() if k != 'prev_rnn_state'))
        self.samples = buffer_from_example(buffer_example, (optim_T, self.buffer_size), share_memory=self.async_)
        self.samples_prev_rnn_state = buffer_from_example(example.prev_rnn_state, (self.buffer_size, ), share_memory=self.async_)

    def append_samples(self, samples):
        with self.rw_lock.write_lock:
            self._async_pull() # get all the updates from other writers

            T, B = get_leading_dims(samples, n_dim=2)
            num_new_sequences = B

            # fill up the buffer all the way down
            if self.t + num_new_sequences >= self.buffer_size:
                num_new_sequences = self.buffer_size - self.t

            B_idxs = np.arange(self.t, self.t + num_new_sequences)
            self.samples[:, B_idxs] = self.SamplesToBuffer(*(v[:, :num_new_sequences] for k, v in samples.items() if k != 'prev_rnn_state'))
            self.samples_prev_rnn_state[B_idxs] = samples.prev_rnn_state[0, :num_new_sequences]
            self._buffer_full = self._buffer_full or (self.t + num_new_sequences) == self.buffer_size
            self.t = (self.t + num_new_sequences) % self.buffer_size
            self._async_push() # send update to other writers and/or readers

    def generate_batches(self, replay_ratio):
        if replay_ratio > 1:
            yield from self._generate_stochastic_batches(replay_ratio)

        else:
            self.clear_buffer() # clear buffer so that new samples are guaranteed to be from new model parameters
            yield from self._generate_deterministic_batches()

    def _generate_deterministic_batches(self):
        cum_sleep_length = 0
        for i in range(self.target_steps):
            while True:
                with self.rw_lock:
                    # get the read lock and start reading the buffers
                    self._async_pull()

                if self.t >= self.optim_B * (i + 1) or self._buffer_full:
                    # stop reading
                    break

                time.sleep(self.sleep_length)
                cum_sleep_length += self.sleep_length if i > 0 else 0

            # now that batches are available, just slice and torchify
            idxs = np.arange(i * self.optim_B, (i + 1) * self.optim_B)

            with self.rw_lock:
                # again capture the read/write lock and slice the samples
                batch_samples = self.samples[:, idxs]
                batch_samples_prev_rnn_state = self.samples_prev_rnn_state[idxs]

            yield torchify_buffer(batch_samples), torchify_buffer(batch_samples_prev_rnn_state), cum_sleep_length

    def _generate_stochastic_batches(self):
        cum_sleep_length = 0
        
        with self.rw_lock:
            # get the read lock and start reading the buffers
            self._async_pull()

        if not self._buffer_full:
            # wait until the buffer is completely full
            print('[INFO] Buffer is not full yet.')
            return

        for _ in range(self.target_steps):
            idxs = np.random.choice(self.buffer_size, self.optim_B)

            with self.rw_lock:
                # slice the samples
                batch_samples = self.samples[:, idxs]
                batch_samples_prev_rnn_state = self.samples_prev_rnn_state[idxs]

            yield torchify_buffer(batch_samples), torchify_buffer(batch_samples_prev_rnn_state), cum_sleep_length

    def clear_buffer(self):
        with self.rw_lock.write_lock:
            self._async_pull()
            self._buffer_full = False
            self.t = 0
            self._async_push()

        
