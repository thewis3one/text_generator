from kivy.app import App
from kivy.core.window import Window
from kivy.properties import ObjectProperty

from kivy.uix.widget import Widget
import GRU
import tensorflow as tf
import pickle

emb_dim = 256
units = 1024
seq_len = 100
BATCH_SIZE = 16


name_of_field = 'rap'
encoding = None
text_len = 200
input_text = 'please stand up'
end_char = '\n'
sequences, ids_from_chars, chars_from_ids = GRU.Load_Preprocess('fields/' + name_of_field + '/text.txt', seq_len, encoding=encoding)
model = GRU.MyModel(len(ids_from_chars.get_vocabulary()), emb_dim, units)
example_input = pickle.load(open('fields/' + name_of_field + '/example', 'rb'))
example_pred = model(example_input)
model.load_weights('fields/' + name_of_field + '/weights.h5')
one_step_model = GRU.OneStep(model, chars_from_ids, ids_from_chars)


def load():
    global sequences
    global ids_from_chars
    global chars_from_ids
    global model
    global one_step_model
    sequences, ids_from_chars, chars_from_ids = GRU.Load_Preprocess('fields/' + name_of_field + '/text.txt', seq_len, encoding=encoding)
    model = GRU.MyModel(len(ids_from_chars.get_vocabulary()), emb_dim, units)
    example_input = pickle.load(open('fields/' + name_of_field + '/example', 'rb'))
    example_pred = model(example_input)
    model.load_weights('fields/' + name_of_field + '/weights.h5')
    one_step_model = GRU.OneStep(model, chars_from_ids, ids_from_chars)


class  MyGridLayout(Widget):
    input = ObjectProperty(None)
    result = ObjectProperty(None)

    def press(self):
        text = self.input.text
        text = GRU.gen_text(text_len, one_step_model, input_text=text, end_char=end_char)
        self.result.text = text

    def radio_button_checked(self, instance, value, field):
        global name_of_field
        if value:
            if name_of_field == field:
                pass
            else:
                name_of_field = field
                load()
        else:
            pass

class GRU_UI(App):
    def build(self):
        return MyGridLayout()


if __name__ == "__main__":
    GRU_UI().run()
