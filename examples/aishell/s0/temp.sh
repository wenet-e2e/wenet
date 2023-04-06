for x in train dev test; do
  cp data/${x}/text data/${x}/text.org
  paste -d " " <(cut -f 1 -d" " data/${x}/text.org) \
  <(cut -f 2- -d" " data/${x}/text.org | tr -d " ") \
  > data/${x}/text
  rm data/${x}/text.org
done
