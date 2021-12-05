#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright [2021-12-04] <sxc19@mails.tsinghua.edu.cn, Xingchen Song>

import pytest

from wenet.utils.file_utils import read_non_lang_symbols


@pytest.mark.parametrize(
    "non_lang_symbol_table_path",
    [
        "test/resources/non-linguistic-symbols.valid",
        "test/resources/non-linguistic-symbols.invalid"
    ]
)
def test_read_non_lang_symbols(non_lang_symbol_table_path):
    path = non_lang_symbol_table_path
    try:
        syms = read_non_lang_symbols(path)
        assert syms[0] == "{~!@#$%^&*()_+`1234567890-=[]|\\\\:;\"'<>,./?}"
        assert syms[1] == "[~!@#$%^&*()_+`1234567890-={}|\\\\:;\"'<>,./?]"
        assert syms[2] == "<~!@#$%^&*()_+`1234567890-={}|\\\\:;\"'[],./?>"
        assert syms[3] == "{qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM}"
        assert syms[4] == "[qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM]"
        assert syms[5] == "<qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM>"
    except Exception as e:
        assert path == "test/resources/non-linguistic-symbols.invalid"
