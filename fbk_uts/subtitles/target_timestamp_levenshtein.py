# Copyright 2021 FBK

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License
import unittest

from examples.speech_to_text.scripts.target_from_source_timestamp_levenshtein import generate_target_timestamp


class TargetTimestampTestCase(unittest.TestCase):
    def test_subtitle_less_segmented(self):
        caption = "I am combining specific types of signals <eob> that mimic how our body in <eob> enter injury to help us regenerate. <eob>"
        subtitle = "Me combino de señales específicas <eob> que imitan cómo responde nuestro cuerpo <eol> y una lesión para ayudarnos a regenerarnos. <eob>"
        time = [[0.21, 2.7], [2.7, 4.68], [4.68, 6.97]]
        timestamp = generate_target_timestamp(caption, subtitle, time)
        self.assertEqual(timestamp, "0.21-2.26425 2.26425-6.97")

        caption = "Just like vaccines instruct <eol> our body to fight disease, <eob> we could insteadruc our immune system <eob> to build tissues and more quickly heal wins. <eob>"
        subtitle = "Así como las vacunas instruían <eol> nuestro cuerpo para combatir enfermedades, <eob> podríamos, en cambio, instruir <eol> nuestro sistema inmune para construir tejidos <eol> y, más rápidamente, curarnos. <eob>"
        time = [[0.05, 3.1], [3.1, 5.86], [5.86, 8.44]]
        timestamp = generate_target_timestamp(caption, subtitle, time)
        self.assertEqual(timestamp, "0.05-4.48 4.48-8.44")

    def test_caption_less_segmented(self):
        caption = "But if you could take a pill or a vaccine <eob> and just like getting over a cold, <eob> you could heal your winds faster. <eob>"
        subtitle = "Pero si pudieras tomar una píldora <eob> o una vacuna, <eob> y, al igual que un resfriado, <eob> podías curar el viento más rápido. <eob>"
        time = [[0.08, 3.52], [3.52, 5.16], [5.16, 6.65]]
        timestamp = generate_target_timestamp(caption, subtitle, time)
        self.assertEqual(timestamp, "0.08-2.9326829268292682 2.9326829268292682-3.801142857142857 3.801142857142857-5.16 5.16-6.65")

        caption = "Today, if we have an operation  or an accident, we' in the hospital for weeks <eob> and often with scars <eol> and painful side effects <eob> of our inability to regenerate  orrerow healthy unred organs. <eob>"
        subtitle = "Hoy en día, si tenemos una operación <eol> o un accidente, <eob> estamos en el hospital por semanas, <eob> y a menudo nos quedan cicatrices y efectos secundarios dolorosos <eob> de nuestra incapacidad de regenerar <eob> o regenerar órganos sanos <eol> y no dañados. <eob>"
        time = [[0.02, 5.11], [5.11, 7.82], [7.82, 12.71]]
        timestamp = generate_target_timestamp(caption, subtitle, time)
        self.assertEqual(
            timestamp,
            "0.02-3.0622988505747135 3.0622988505747135-5.11 5.11-7.82 7.82-10.132837837837839 10.132837837837839-12.71"
        )

    def test_caption_subtitle_equal_segmentation(self):
        caption = "I work to create materials <eol> that instruct our immune system <eob> to give us the signals <eol> to grow new tissues. <eob>"
        subtitle = "Trabajo para crear materiales <eob> que instruyan nuestro sistema inmune <eol> para darnos las señales, para cultivar nuevos tejidos. <eob>"
        time = [[0.19, 3.67], [3.67, 6.07]]
        timestamp = generate_target_timestamp(caption, subtitle, time)
        self.assertEqual(timestamp, "0.19-1.93 1.93-6.07")

        caption = "I work to create materials <eol> that instruct our immune system <eob> to give us the signals <eol> to grow <eob> new tissues. <eob>"
        subtitle = "Trabajo <eob> para crear materiales <eob> que instruyan nuestro sistema inmune <eol> para darnos las señales, para cultivar nuevos tejidos. <eob>"
        time = [[0.19, 3.67], [3.67, 5], [5, 6.07]]
        timestamp = generate_target_timestamp(caption, subtitle, time)
        self.assertEqual(timestamp, "0.19-0.61 0.61-1.8699999999999999 1.8699999999999999-6.07")

    def test_caption_extra_eob(self):
        caption = 'And that was a time when I first ask myself, "Why? <eob>'
        subtitle = 'Und das war die Zeit, <eol> in der ich mich zum ersten Mal fragte: <eob> "Warum? <eob>'
        time = [[0.08, 3.44]]
        timestamp = generate_target_timestamp(caption, subtitle, time)
        self.assertEqual("0.08-3.088955223880597 3.088955223880597-3.44", timestamp)


if __name__ == '__main__':
    unittest.main()
