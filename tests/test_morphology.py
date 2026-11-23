from atomzero.morphology import looks_turkish, skeleton, variants


def test_turkish_skeleton_handles_mutation_and_suffixes():
    assert skeleton("kitabı") == "kitap"
    assert skeleton("kitabın") == "kitap"
    assert skeleton("kitaplar") == "kitap"
    assert skeleton("ağacı") == "ağaç"
    assert skeleton("sokağı") == "sokak"
    assert skeleton("rengi") == "renk"
    assert skeleton("kanadı") == "kanat"
    assert skeleton("evler") == "ev"
    assert skeleton("sular") == "su"
    assert skeleton("yollar") == "yol"
    assert skeleton("şehir") == "şehr"
    assert skeleton("şehri") == "şehr"
    assert skeleton("burun") == "burn"
    assert skeleton("burnu") == "burn"
    assert skeleton("ağız") == "ağz"
    assert skeleton("ağzı") == "ağz"
    assert skeleton("isim") == "ism"
    assert skeleton("ismi") == "ism"
    assert skeleton("kayıt") == "kayd"
    assert skeleton("kaydı") == "kayd"
    # Common words that merely end in a suffix-like sequence must survive.
    assert skeleton("hasta") == "hasta"
    assert skeleton("masa") == "masa"


def test_turkish_detection_and_variants():
    assert looks_turkish("Kitap kapağı ne renk?")
    assert looks_turkish("Ev eski mi?")
    assert looks_turkish("Burun?")
    assert not looks_turkish("What is the capital of France?")
    assert not looks_turkish("her book")
    assert "kitab" in variants("kitap")
    assert "ağac" in variants("ağaç")
    assert "sokağ" in variants("sokak")
    assert "şehr" in variants("şehir")
