"""Check transport integrity before accepting or extracting a result archive.
Author: James Edward Ball.
Scientific admission is performed separately by the frozen result verifiers.
"""
from pathlib import PurePosixPath
import hashlib
import json
import tarfile


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_archive(path):
    with tarfile.open(path) as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        require(len(names) == len(set(names)), 'Duplicate archive member')
        for member in members:
            name = PurePosixPath(member.name)
            require(member.isfile() and not name.is_absolute()
                    and '..' not in name.parts and str(name) == member.name,
                    'Unsafe archive member: ' + member.name)
        require('collection-manifest.json' in names, 'Missing collection manifest')
        receipt = json.load(archive.extractfile('collection-manifest.json'))
        expected = receipt['files']
        require(receipt['ready'] is True and isinstance(expected, dict) and expected,
                'Incomplete collection receipt')
        require(set(names) == set(expected) | {'collection-manifest.json'}
                and 'collection-manifest.json' not in expected,
                'Archive inventory differs from receipt')
        for member in members:
            if member.name == 'collection-manifest.json':
                continue
            digest = hashlib.sha256()
            with archive.extractfile(member) as stream:
                for chunk in iter(lambda: stream.read(8 << 20), b''):
                    digest.update(chunk)
            record = expected[member.name]
            require(digest.hexdigest() == record['sha256']
                    and member.size == record['bytes'],
                    'Archive content differs from receipt: ' + member.name)
    return receipt
