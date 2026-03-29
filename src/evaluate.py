import os

from codec import encode

K = 223  # encoded symbols


def evaluate_decoder(decoder, channel_fn, num_samples=1000, return_details=False):
    success = 0

    for _ in range(num_samples):
        msg = os.urandom(K)
        codeword = encode(msg)
        noisy, _ = channel_fn(codeword)

        decoded = decoder.decode(noisy)
        if decoded == msg:
            success += 1

    fsr = success / num_samples

    if return_details:
        return {"fsr": fsr, "success": success, "total": num_samples}
    return fsr


def compare_decoders(decoders: dict, channel_fn, num_samples=1000):
    """Compare multiple decoders on the same noisy data."""
    successes = {key: 0 for key in decoders}

    for _ in range(num_samples):
        msg = os.urandom(K)
        codeword = encode(msg)
        noisy, erase_pos = channel_fn(codeword)

        for key in decoders.keys():
            if decoders[key].decode(noisy=noisy, erase_pos=erase_pos, original=codeword) == msg:
                successes[key] += 1

    return {key: count / num_samples for key, count in successes.items()}
