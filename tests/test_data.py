import pytest

from facerec.data import parse_pairs, person_of


def test_pairs_file_gives_same_and_different_pairs_with_folds():
    text = "2\t1\nAnn\t1\t2\nBob\t1\tCal\t3\nDee\t2\t4\nEve\t1\tFay\t1\n"
    pairs = parse_pairs(text)
    assert [(p.first, p.second, p.same, p.fold) for p in pairs] == [
        ("Ann/Ann_0001.jpg", "Ann/Ann_0002.jpg", True, 0),
        ("Bob/Bob_0001.jpg", "Cal/Cal_0003.jpg", False, 0),
        ("Dee/Dee_0002.jpg", "Dee/Dee_0004.jpg", True, 1),
        ("Eve/Eve_0001.jpg", "Fay/Fay_0001.jpg", False, 1),
    ]


def test_pairs_file_with_missing_lines_is_rejected():
    with pytest.raises(ValueError):
        parse_pairs("1\t2\nAnn\t1\t2\n")


def test_person_comes_from_the_folder_name():
    assert person_of("George_W_Bush/George_W_Bush_0001.jpg") == "George_W_Bush"
