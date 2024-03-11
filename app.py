from flask import Flask, render_template, request
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np
from keras.models import load_model  # TensorFlow is required for Keras to work

app = Flask(__name__)

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("model/converted_keras_MO106.h5", compile=False)

# Load the labels
class_names = open("model/labels_MO106.txt", "r").readlines()

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

@app.route('/', methods=['GET'])
def index_page():
    return render_template('index.html')

@app.route('/class')
def class_page():
    return render_template('classifier.html')

# Dictionary mapping labels to additional information
label_info = {  
    'Agaricus augustus':{'family':'Agaricaceae','edibility':'Edible','weblink':'https://en.wikipedia.org/wiki/Agaricus_augustus','region':'Southeast Asia'},
    'Agaricus xanthodermus':{'family':'Agaricaceae','edibility':'Edible','weblink':'https://en.wikipedia.org/wiki/Agaricus_xanthodermus','region':'Southeast Asia'},
    'Amanita amerirubescens':{'family':'Amanitaceae','edibility':'Edible','weblink':'https://en.wikipedia.org/wiki/Blusher','region':'Southeast Asia'},
    'Amanita augusta':{'family':'Amanitaceae','edibility':'Edible','weblink':'https://en.wikipedia.org/wiki/Amanita_augusta','region':'Southeast Asia'},
    'Amanita brunnescens':{'family':'Amanitaceae','edibility':'Non-edible','weblink':'https://en.wikipedia.org/wiki/Amanita_brunnescens','region':'Southeast Asia'},
    'Amanita calyptroderma':{'family':'Amanitaceae','edibility':'Non-edible','weblink':'https://en.wikipedia.org/wiki/Amanita_calyptroderma ','region':'Southeast Asia'},
    'Amanita flavoconia':{'family':'Amanitaceae','edibility':'Non-edible','weblink':'https://en.wikipedia.org/wiki/Amanita_flavoconia ','region':'Southeast Asia'},
    'Amanita muscaria':{'family':'Amanitaceae','edibility':'Non-edible','weblink':'https://en.wikipedia.org/wiki/Amanita_muscaria ','region':'Worldwide'},
    'Amanita persicina':{'family':'Amanitaceae','edibility':'Edible','weblink':'https://en.wikipedia.org/wiki/Amanita_persicina ','region':'Southeast Asia'},
    'Amanita phalloides':{'family':'Amanitaceae','edibility':'Non-edible','weblink':'https://en.wikipedia.org/wiki/Amanita_phalloides ','region':'Worldwide'},
    'Amanita velosa':{'family':'Amanitaceae','edibility':'Non-edible','weblink':'https://en.wikipedia.org/wiki/Amanita_velosa ','region':'Southeast Asia'},
    'Armillaria mellea':{'family':'Physalacriaceae','edibility':'Edible','weblink':'https://en.wikipedia.org/wiki/Armillaria_mellea ','region':'Worldwide'},
    'Armillaria tabescens':{'family':'Physalacriaceae','edibility':'Edible','weblink':'https://en.wikipedia.org/wiki/Armillaria_tabescens ','region':'Worldwide'},
    'Artomyces pyxidatus':{'family':'Physalacriaceae','edibility':'Edible','weblink':'https://en.wikipedia.org/wiki/Artomyces_pyxidatus ','region':'Southeast Asia'},
    'Bolbitius titubans':{'family':'Bolbitiaceae','edibility':'Edible','weblink':'https://en.wikipedia.org/wiki/Bolbitius_titubans ','region':'Southeast Asia'},
    'Boletus pallidus':{'family':'Boletaceae','edibility':'Edible','weblink':'https://en.wikipedia.org/wiki/Boletus_pallidus ','region':'Southeast Asia'},
    'Boletus rex-veris':{'family':'Boletaceae','edibility':'Edible','weblink':'https://en.wikipedia.org/wiki/Boletus_rex-veris ','region':'Southeast Asia'},
    'Cantharellus californicus':{'family':'Cantharellaceae','edibility':'Edible','weblink':'https://en.wikipedia.org/wiki/Cantharellus_californicus ','region':'North America'},
    'Cantharellus cinnabarinus':{'family':'Cantharellaceae','edibility':'Edible','weblink':'https://en.wikipedia.org/wiki/Cantharellus_cinnabarinus ','region':'Southeast Asia'},
    'Cerioporus squamosus':{'family':'Polyporaceae','edibility':'Edible','weblink':'https://en.wikipedia.org/wiki/Cerioporus_squamosus ','region':'Worldwide'},
    'Chlorophyllum brunneum':{'family':'Agaricaceae','edibility':'Edible','weblink':'https://en.wikipedia.org/wiki/Shaggy_parasol ','region':'Worldwide'},
    'Chlorophyllum molybdites':{'family':'Agaricaceae','edibility':'Edible','weblink':'https://en.wikipedia.org/wiki/Chlorophyllum_molybdites ','region':'Worldwide'},
    'Clitocybe nuda':{'family':'Tricholomataceae','edibility':'Edible','weblink':'https://en.wikipedia.org/wiki/Clitocybe_nuda ','region':'Worldwide'},
    'Coprinellus micaceus':{'family':'Psathyrellaceae','edibility':'Edible','weblink':'https://en.wikipedia.org/wiki/Coprinellus_micaceus ','region':'Worldwide'},
    'Coprinopsis lagopus':{'family':'Psathyrellaceae','edibility':'Edible','weblink':'https://en.wikipedia.org/wiki/Coprinopsis_lagopus ','region':'Worldwide'},
    'Coprinus comatus':{'family':'Psathyrellaceae','edibility':'Edible','weblink':'https://en.wikipedia.org/wiki/Coprinus_comatus ','region':'Worldwide'},
    'Crucibulum laeve':{'family':'Nidulariaceae','edibility':'Edible','weblink':'https://en.wikipedia.org/wiki/Crucibulum ','region':'Worldwide'},
    'Cryptoporus volvatus':{'family':'	Polyporaceae','edibility':'Edible','weblink':'https://en.wikipedia.org/wiki/Cryptoporus_volvatus ','region':'Southeast Asia'},
    'Daedaleopsis confragosa':{'family':'Polyporaceae','edibility':'Edible','weblink':'https://en.wikipedia.org/wiki/Daedaleopsis_confragosa ','region':'Worldwide'},
    'Entoloma abortivum':{'family':'Entolomataceae','edibility':'Non-edible','weblink':'https://en.wikipedia.org/wiki/Entoloma_abortivum ','region':'Worldwide'},
    'Flammulina velutipes':{'family':'Physalacriaceae','edibility':'Edible','weblink':'https://en.wikipedia.org/wiki/Flammulina_velutipes ','region':'Worldwide'},
    'Fomitopsis mounceae':{'family':'Fomitopsidaceae','edibility':'Non-edible','weblink':'https://en.wikipedia.org/wiki/Fomitopsis_mounceae ','region':'Worldwide'},
    'Galerina marginata':{'family':'Hymenogastraceae','edibility':'Non-edible','weblink':'https://en.wikipedia.org/wiki/Galerina_marginata ','region':'Worldwide'},
    'Ganoderma applanatum':{'family':'Ganodermataceae','edibility':'Non-edible','weblink':'https://en.wikipedia.org/wiki/Ganoderma_applanatum ','region':'Worldwide'},
    'Ganoderma curtisii':{'family':'Ganodermataceae','edibility':'Non-edible','weblink':'https://en.wikipedia.org/wiki/Ganoderma_curtisii ','region':'Worldwide'},
    'Ganoderma oregonense':{'family':'Ganodermataceae','edibility':'Non-edible','weblink':'https://www.mushroomexpert.com/ganoderma_oregonense.html ','region':'North America'},
    'Ganoderma tsugae':{'family':'Ganodermataceae','edibility':'Non-edible','weblink':'https://en.wikipedia.org/wiki/Ganoderma_tsugae ','region':'North America'},
    'Gliophorus psittacinus':{'family':'Hygrophoraceae','edibility':'Edible','weblink':'https://en.wikipedia.org/wiki/Gliophorus_psittacinus ','region':'Southeast Asia'},
    'Gloeophyllum sepiarium':{'family':'Gloeophyllaceae','edibility':'Non-edible','weblink':'https://en.wikipedia.org/wiki/Gloeophyllum_sepiarium ','region':'Worldwide'},
    'Grifola frondosa':{'family':'Meripilaceae','edibility':'Edible','weblink':'https://en.wikipedia.org/wiki/Grifola_frondosa ','region':'Worldwide'},
    'Gymnopilus luteofolius':{'family':'Hymenogastraceae','edibility':'Edible','weblink':'https://en.wikipedia.org/wiki/Gymnopilus_luteofolius ','region':'Southeast Asia'},
    'Hericium coralloides':{'family':'Hericiaceae','edibility':'Edible','weblink':'https://en.wikipedia.org/wiki/Hericium_coralloides ','region':'Worldwide'},
    'Hericium erinaceus':{'family':'Hericiaceae','edibility':'Edible','weblink':'https://en.wikipedia.org/wiki/Hericium_erinaceus ','region':'Worldwide'},
    'Hygrophoropsis aurantiaca':{'family':'Hygrophoropsidaceae','edibility':'Edible','weblink':'https://en.wikipedia.org/wiki/Hygrophoropsis_aurantiaca ','region':'Worldwide'},
    'Hypholoma fasciculare':{'family':'Strophariaceae','edibility':'Edible','weblink':'https://en.wikipedia.org/wiki/Hypholoma_fasciculare ','region':'Worldwide'},
    'Hypholoma lateritium':{'family':'Strophariaceae','edibility':'Edible','weblink':'https://en.wikipedia.org/wiki/Hypholoma_lateritium ','region':'Worldwide'},
    'Hypomyces lactifluorum':{'family':'Hypocreaceae','edibility':'Non-edible','weblink':'https://en.wikipedia.org/wiki/Hypomyces_lactifluorum ','region':'Worldwide'},
    'Ischnoderma resinosum':{'family':'Fomitopsidaceae','edibility':'Non-edible','weblink':'https://en.wikipedia.org/wiki/Ischnoderma_resinosum ','region':'Worldwide'},
    'Laccaria ochropurpurea':{'family':'Hydnangiaceae','edibility':'Edible','weblink':'https://en.wikipedia.org/wiki/Laccaria_ochropurpurea ','region':'Worldwide'},
    'Lacrymaria lacrymabunda':{'family':'Psathyrellaceae','edibility':'Edible','weblink':'https://en.wikipedia.org/wiki/Lacrymaria_lacrymabunda ','region':'Worldwide'},
    'Lactarius indigo':{'family':'Russulaceae','edibility':'Edible','weblink':'https://en.wikipedia.org/wiki/Lactarius_indigo ','region':'Worldwide'},
    'Laetiporus sulphureus':{'family':'Fomitopsidaceae','edibility':'Edible','weblink':'https://en.wikipedia.org/wiki/Laetiporus_sulphureus ','region':'Worldwide'},
    'Laricifomes officinalis':{'family':'Laricifomitaceae','edibility':'Non-edible','weblink':'https://en.wikipedia.org/wiki/Laricifomes_officinalis ','region':'Worldwide'},
    'Leratiomyces ceres':{'family':'Strophariaceae','edibility':'Non-edible','weblink':'https://en.wikipedia.org/wiki/Leratiomyces_ceres ','region':'Southeast Asia'},
    'Leucoagaricus americanus':{'family':'Agaricaceae','edibility':'Edible','weblink':'https://en.wikipedia.org/wiki/Leucoagaricus_americanus ','region':'North America'},
    'Leucoagaricus leucothites':{'family':'Agaricaceae','edibility':'Edible','weblink':'https://en.wikipedia.org/wiki/Leucoagaricus_leucothites ','region':'Southeast Asia'},
    'Lycogala epidendrum':{'family':'Tubiferaceae','edibility':'Non-edible','weblink':'https://en.wikipedia.org/wiki/Lycogala_epidendrum ','region':'Worldwide'},
    'Lycoperdon perlatum':{'family':'Agaricaceae','edibility':'Edible','weblink':'https://en.wikipedia.org/wiki/Lycoperdon_perlatum ','region':'Worldwide'},
    'Lycoperdon pyriforme':{'family':'Lycoperdaceae','edibility':'Edible','weblink':'https://en.wikipedia.org/wiki/Apioperdon ','region':'Worldwide'},
    'Mycena haematopus':{'family':'Mycenaceae','edibility':'Non-edible','weblink':'https://en.wikipedia.org/wiki/Mycena_haematopus ','region':'Worldwide'},
    'Mycena leaiana':{'family':'Mycenaceae','edibility':'Non-edible','weblink':'https://en.wikipedia.org/wiki/Mycena_leaiana ','region':'Worldwide'},
    'Omphalotus illudens':{'family':'Omphalotaceae','edibility':'Non-edible','weblink':'https://en.wikipedia.org/wiki/Omphalotus_illudens ','region':'Europe, North America, Asia'},
    'Omphalotus olivascens':{'family':'Omphalotaceae','edibility':'Non-edible','weblink':'https://en.wikipedia.org/wiki/Omphalotus_olivascens ','region':'Europe, Asia'},
    'Panaeolina foenisecii':{'family':'Bolbitiaceae','edibility':'Non-edible','weblink':'https://en.wikipedia.org/wiki/Panaeolus_foenisecii ','region':'Southeast Asia'},
    'Panaeolus cinctulus':{'family':'Bolbitiaceae','edibility':'Edible','weblink':'https://en.wikipedia.org/wiki/Panaeolus_cinctulus ','region':'Southeast Asia'},
    'Panaeolus papilionaceus':{'family':'Bolbitiaceae','edibility':'Edible','weblink':'https://en.wikipedia.org/wiki/Panaeolus_papilionaceus ','region':'Southeast Asia'},
    'Panellus stipticus':{'family':'Mycenaceae','edibility':'Edible','weblink':'https://en.wikipedia.org/wiki/Panellus_stipticus ','region':'Southeast Asia'},
    'Phaeolus schweinitzii':{'family':'Fomitopsidaceae','edibility':'Non-edible','weblink':'https://en.wikipedia.org/wiki/Phaeolus_schweinitzii ','region':'Worldwide'},
    'Phlebia tremellosa':{'family':'Meruliaceae','edibility':'Non-edible','weblink':'https://en.wikipedia.org/wiki/Phlebia_tremellosa ','region':'Worldwide'},
    'Phyllotopsis nidulans':{'family':'Phyllotopsidaceae','edibility':'Edible','weblink':'https://en.wikipedia.org/wiki/Phyllotopsis_nidulans ','region':'Southeast Asia'},
    'Pleurotus ostreatus':{'family':'Pleurotaceae','edibility':'Edible','weblink':'https://en.wikipedia.org/wiki/Pleurotus_ostreatus ','region':'Worldwide'},
    'Pleurotus pulmonarius':{'family':'Pleurotaceae','edibility':'Edible','weblink':'https://en.wikipedia.org/wiki/Pleurotus_pulmonarius ','region':'Southeast Asia'},
    'Pluteus cervinus':{'family':'Pluteaceae','edibility':'Edible','weblink':'https://en.wikipedia.org/wiki/Pluteus_cervinus ','region':'Worldwide'},
    'Psathyrella candolleana':{'family':'Psathyrellaceae','edibility':'Non-edible','weblink':'https://en.wikipedia.org/wiki/Candolleomyces_candolleanus ','region':'Worldwide'},
    'Pseudohydnum gelatinosum':{'family':'incertae sedis','edibility':'Edible','weblink':'https://en.wikipedia.org/wiki/Pseudohydnum_gelatinosum ','region':'Southeast Asia'},
    'Psilocybe allenii':{'family':'Hymenogastraceae','edibility':'Hallucinogenic','weblink':'https://en.wikipedia.org/wiki/Psilocybe_allenii ','region':'Southeast Asia'},
    'Psilocybe aztecorum':{'family':'Hymenogastraceae','edibility':'Hallucinogenic','weblink':'https://en.wikipedia.org/wiki/Psilocybe_aztecorum ','region':'Central America, Mexico'},
    'Psilocybe azurescens':{'family':'Hymenogastraceae','edibility':'Hallucinogenic','weblink':'https://en.wikipedia.org/wiki/Psilocybe_azurescens ','region':'North America, Europe'},
    'Psilocybe caerulescens':{'family':'Hymenogastraceae','edibility':'Hallucinogenic','weblink':'https://en.wikipedia.org/wiki/Psilocybe_caerulescens ','region':'Southeast Asia'},
    'Psilocybe cubensis':{'family':'Hymenogastraceae','edibility':'Hallucinogenic','weblink':'https://en.wikipedia.org/wiki/Psilocybe_cubensis ','region':'Worldwide'},
    'Psilocybe cyanescens':{'family':'Hymenogastraceae','edibility':'Hallucinogenic','weblink':'https://en.wikipedia.org/wiki/Psilocybe_cyanescens ','region':'Europe, North America'},
    'Psilocybe muliercula':{'family':'Hymenogastraceae','edibility':'Hallucinogenic','weblink':'https://en.wikipedia.org/wiki/Psilocybe_muliercula ','region':'Southeast Asia'},
    'Psilocybe neoxalapensis':{'family':'Hymenogastraceae','edibility':'Hallucinogenic','weblink':'https://en.wikipedia.org/wiki/Psilocybe_neoxalapensis ','region':'Mexico, Central America'},
    'Psilocybe ovoideocystidiata':{'family':'Hymenogastraceae','edibility':'Hallucinogenic','weblink':'https://en.wikipedia.org/wiki/Psilocybe_ovoideocystidiata ','region':'Europe, North America'},
    'Psilocybe pelliculosa':{'family':'Hymenogastraceae','edibility':'Hallucinogenic','weblink':'https://en.wikipedia.org/wiki/Psilocybe_pelliculosa ','region':'Southeast Asia'},
    'Psilocybe zapotecorum':{'family':'Hymenogastraceae','edibility':'Hallucinogenic','weblink':'https://en.wikipedia.org/wiki/Psilocybe_zapotecorum ','region':'Central America, Mexico'},
    'Retiboletus ornatipes':{'family':'Boletaceae','edibility':'Edible','weblink':'https://en.wikipedia.org/wiki/Retiboletus_ornatipes ','region':'Southeast Asia'},
    'Sarcomyxa serotina':{'family':'Sarcomyxaceae','edibility':'Edible','weblink':'https://en.wikipedia.org/wiki/Sarcomyxa_serotina ','region':'Europe, Asia, North America'},
    'Schizophyllum commune':{'family':'Schizophyllaceae','edibility':'Edible','weblink':'https://en.wikipedia.org/wiki/Schizophyllum_commune ','region':'Worldwide'},
    'Stereum ostrea':{'family':'Stereaceae','edibility':'Edible','weblink':'https://en.wikipedia.org/wiki/Stereum_ostrea ','region':'Worldwide'},
    'Stropharia ambigua':{'family':'Strophariaceae','edibility':'Edible','weblink':'https://en.wikipedia.org/wiki/Stropharia_ambigua ','region':'Southeast Asia'},
    'Stropharia rugosoannulata':{'family':'Strophariaceae','edibility':'Edible','weblink':'https://en.wikipedia.org/wiki/Stropharia_rugosoannulata ','region':'Southeast Asia'},
    'Suillus americanus':{'family':'Suillaceae','edibility':'Edible','weblink':'https://en.wikipedia.org/wiki/Suillus_americanus ','region':'North America'},
    'Suillus luteus':{'family':'Suillaceae','edibility':'Edible','weblink':'https://en.wikipedia.org/wiki/Suillus_luteus ','region':'North America'},
    'Suillus spraguei':{'family':'Suillaceae','edibility':'Edible','weblink':'https://en.wikipedia.org/wiki/Suillus_spraguei ','region':'North America'},
    'Tapinella atrotomentosa':{'family':'Tapinellaceae','edibility':'Edible','weblink':'https://en.wikipedia.org/wiki/Tapinella_atrotomentosa  ','region':'Southeast Asia'},
    'Trametes betulina':{'family':'Polyporaceae','edibility':'Edible','weblink':'https://en.wikipedia.org/wiki/Trametes_betulina ','region':'Worldwide'},
    'Trametes gibbosa':{'family':'Polyporaceae','edibility':'Edible','weblink':'https://en.wikipedia.org/wiki/Trametes_gibbosa ','region':'Worldwide'},
    'Trametes versicolor':{'family':'Polyporaceae','edibility':'Edible','weblink':'https://en.wikipedia.org/wiki/Trametes_versicolor ','region':'Worldwide'},
    'Trichaptum biforme':{'family':'incertae sedis','edibility':'Edible','weblink':'https://en.wikipedia.org/wiki/Trichaptum_biforme ','region':'Worldwide'},
    'Tricholoma murrillianum':{'family':'Tricholomataceae','edibility':'Edible','weblink':'https://en.wikipedia.org/wiki/Tricholoma_murrillianum ','region':'North America'},
    'Tricholomopsis rutilans':{'family':'Tricholomataceae','edibility':'Edible','weblink':'https://en.wikipedia.org/wiki/Tricholomopsis_rutilans ','region':'Southeast Asia'},
    'Tubaria furfuracea':{'family':'Tubariaceae','edibility':'Edible','weblink':'https://en.wikipedia.org/wiki/Tubaria_furfuracea ','region':'Southeast Asia'},
    'Tylopilus felleus':{'family':'Boletaceae','edibility':'Non-edible','weblink':'https://en.wikipedia.org/wiki/Tylopilus_felleus ','region':'Worldwide'},
    'Tylopilus rubrobrunneus':{'family':'Boletaceae','edibility':'Edible','weblink':'https://en.wikipedia.org/wiki/Tylopilus_rubrobrunneus ','region':'Southeast Asia'},
    'Volvopluteus gloiocephalus':{'family':'Pluteaceae','edibility':'Edible','weblink':'https://en.wikipedia.org/wiki/Volvopluteus_gloiocephalus ','region':'Southeast Asia'}
}

@app.route('/predict', methods=['POST'])
def predict():
    imagefile= request.files['imagefile']
    image_path = "./static/test/predict.jpg"
    imagefile.save(image_path)

    # Replace this with the path to your image
    image = Image.open(image_path).convert("RGB")

    # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    imagefile = image
    imagefile.save(image_path)

    # turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # Predicts the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    label = class_names[index]
    confidence = prediction[0][index]
    family='family'
    edibility='edibility'
    weblink='weblink'
    region='region'

    # Get the label information based on the predicted label
    label_info_dict = label_info.get(label.strip())

    # Extract information for the predicted label
    if label_info_dict:
        family = label_info_dict.get('family', 'Family info not available')
        edibility = label_info_dict.get('edibility', 'Edibility info not available')
        weblink = label_info_dict.get('weblink', '#')
        region = label_info_dict.get('region', 'Region info not available')

    return render_template('classifier.html', label=label, confidence=confidence, img_path=image_path, family=family, edibility=edibility, weblink=weblink, region=region)


if __name__ == '__main__':
    app.run(port=3000, debug=True)

    