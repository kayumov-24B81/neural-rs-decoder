"""Load real network payloads form a pcap file as RS messages.

Parses the classic libpcap format directly (no external depdendencies):
a 24-byte global header followed by records, each with a 16-byte header
and the raw packet bytes. Transport-layer payloads (TCP/UDP over IPv4 or
IPv6, Ethernet link layer) are concatenated and sliced into fixed-size messages.

PPPoE-encapsulation traffic is not unwrapped; such packets are skipped.
"""

import struct

# pcap magic numbers -> (struct endianness prefix)
_MAGICS = {
    b"\xd4\xc3\xb2\xa1": "<",  # little-endian, microsecond
    b"\xa1\xb2\xc3\xd4": ">",  # big-endian, microsecond
    b"\x4d\x3c\xb2\xa1": "<",  # little-endian, nanosecond
    b"\xa1\xb2\x3c\x4d": ">",  # big-endian, nanosecond
}

_LINKTYPE_ETHERNET = 1
_ETHERTYPE_IPV4 = 0x0800
_ETHERTYPE_IPV6 = 0x86DD
_IP_PROTO_TCP = 6
_IP_PROTO_UDP = 17


def _extract_payload(pkt: bytes) -> bytes:
    """Return the TCP/UDP payload of one Ethernet frame, or b'' if none."""
    if len(pkt) < 14:
        return b""
    ethertype = struct.unpack(">H", pkt[12:14])[0]
    l3 = pkt[14:]

    if ethertype == _ETHERTYPE_IPV4:
        if len(l3) < 20:
            return b""
        ihl = (l3[0] & 0x0F) * 4
        proto = l3[9]
        l4 = l3[ihl:]
    elif ethertype == _ETHERTYPE_IPV6:
        if len(l3) < 40:
            return b""
        proto = l3[6]
        l4 = l3[40:]
    else:
        return b""

    if proto == _IP_PROTO_TCP and len(l4) >= 20:
        data_off = ((l4[12] >> 4) & 0x0F) * 4
        return bytes(l4[data_off:])
    if proto == _IP_PROTO_UDP and len(l4) >= 8:
        return bytes(l4[8:])
    return b""


def load_pcap_messages(path: str, msg_len: int = 223) -> list:
    """Extract transport payloads from a classic pcap and slice into messages.

    All non-empty TCP/UDP payloads are concatenated into one byte stream,
    then cut into fixed-size blocks of msg_len bytes; a trailing partial
    block is discarded.

    Returns a list of msg_len-byte `bytes` objects. Raises ValueError on an
    unrecognized format, an unsupported link type, or if the pcap yields
    less than one full block.
    """
    with open(path, "rb") as f:
        data = f.read()

    if len(data) < 24:
        raise ValueError(f"pcap '{path}' too short to contain a global header.")

    endian = _MAGICS.get(data[:4])
    if endian is None:
        raise ValueError(
            f"pcap '{path}' has unrecognized magic {data[:4].hex()}; "
            f"only the classic libpcap format is supported."
        )

    link_type = struct.unpack(endian + "I", data[20:24])[0]
    if link_type != _LINKTYPE_ETHERNET:
        raise ValueError(
            f"pcap '{path}' link type {link_type} is unsupported "
            f"(only Ethernet, type {_LINKTYPE_ETHERNET})."
        )

    stream = bytearray()
    off = 24
    rec_header = endian + "IIII"
    while off + 16 <= len(data):
        _, _, incl_len, _ = struct.unpack(rec_header, data[off : off + 16])
        off += 16
        if off + incl_len > len(data):
            break  # truncated trailing record
        stream.extend(_extract_payload(data[off : off + incl_len]))
        off += incl_len

    n_blocks = len(stream) // msg_len
    if n_blocks == 0:
        raise ValueError(
            f"pcap '{path}' yielded only {len(stream)} payload bytes, "
            f"need at least {msg_len} for one block."
        )
    return [bytes(stream[i * msg_len : (i + 1) * msg_len]) for i in range(n_blocks)]
