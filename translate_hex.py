import struct

# Sample binary data (in hex) from the registry
data = bytes.fromhex(
    "010004805c0000006c00000000000000140000000200480003000000000018001f00000001020000000000052000000020020000000014001f000000010100000000000504000000000014001f0000000101000000000005120000000102000000000005200000002002000000"
)

# Read ACL header
acl_revision, acl_size, ace_count, sbz2 = struct.unpack("<BxHHL", data[:12])
print(f"ACL Revision: {acl_revision}, ACL Size: {acl_size}, ACE Count: {ace_count}")

# Read ACEs
offset = 12
for i in range(ace_count):
    ace_type, ace_flags, ace_size = struct.unpack("<BBH", data[offset : offset + 4])
    access_mask = struct.unpack("<L", data[offset + 4 : offset + 8])[0]
    sid_start = offset + 8
    sid_length = ace_size - 8
    sid = data[sid_start : sid_start + sid_length]

    # Decode the SID
    sid_revision, sid_sub_authority_count = struct.unpack("<BB", sid[:2])
    sid_authority = struct.unpack(">Q", b"\x00\x00" + sid[2:8])[0]
    sid_sub_authorities = struct.unpack(
        "<" + "L" * sid_sub_authority_count, sid[8 : 8 + 4 * sid_sub_authority_count]
    )

    sid_str = f"S-{sid_revision}-{sid_authority}"
    for sub_authority in sid_sub_authorities:
        sid_str += f"-{sub_authority}"

    print(
        f"ACE Type: {ace_type}, ACE Flags: {ace_flags}, ACE Size: {ace_size}, Access Mask: {access_mask}, SID: {sid_str}"
    )

    offset += ace_size
