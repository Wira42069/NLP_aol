## Error Analysis Examples

### Category: Ambiguous/Multi-Emotion

- **Review:** "Kualitas oke sesuai harga, tapi keren parah ??????"
  - **True:** love | **Predicted:** happy (71%)

- **Review:** "Untuk jar bahannya bagus. bahan2 yg lain lumayan. Namun untuk tutup jar pita tlg diperhatikan lubangnya terlalu besar. gampang ditarik smp lepas. tp terbalik dgn tutupnya stick yg kekecilan lobangnya. stick plastik g bisa masuk. akhirnya hrs diganti tutup stick jd tutup pita. tp y pe er ngerjainnya krn udh keiket kecil2 gt kan. maaf agak panjang reviewnya. untuk yang lain bagus."
  - **True:** fear | **Predicted:** happy (41%)

- **Review:** "barang tidak diterima, tidak ada yang bertanggung jawab, pertama kali belanja di sini langsung dikecewain,,,"
  - **True:** fear | **Predicted:** sadness (47%)

### Category: Annotation Noise

- **Review:** "Highly recomend ????Mantap seller ini pengiriman super kilat cuma 1x24jam sampai,ada yg kurang kirim saya wa cepat dibalas dan langsung dikirim kekurangannya,ramah,respon sangat baik."
  - **True:** happy | **Predicted:** love (13%)

- **Review:** "Pesan 3 buah XXXL ukuran lingkar dada 52 cm, yg datang 43, 44 dan 45 cm. Jauh bos ukurannya, cuma kepake buat lap motor aja ini mah"
  - **True:** anger | **Predicted:** sadness (34%)

- **Review:** "Jahitan rapi. Modelnya cantik ??. Nyaman dipakai. Bahannya adem. Tidak menerawang."
  - **True:** love | **Predicted:** happy (84%)

### Category: Slang/Spelling Outliers

- **Review:** "ga sesuai permintaan...udh jelas2 dikasi keterangan orderannya dan di chat biar inget eeh...malah barang yg datang warna tdk sesuai dengan jumlah warna yg diinginkan. pengen cepet ngirim aja ini lapaknya. klo ga punya stok jgn di approve."
  - **True:** anger | **Predicted:** fear (46%)

- **Review:** "Bangke sampe Rumah mati semua. Jangan beli disini."
  - **True:** anger | **Predicted:** fear (59%)

- **Review:** "l2 r2 trigger nya ampas macam ***"
  - **True:** anger | **Predicted:** happy (33%)
