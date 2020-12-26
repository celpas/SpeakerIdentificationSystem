import logging
from flask import Flask
from flask import render_template, make_response, request, url_for, redirect, flash, jsonify
from web_app.web_app_utils import *
from controllers.audio_manager import AudioManager
from controllers.sentences_manager import SentencesManager
import re
import time
from hashlib import sha256

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

audio_manager = AudioManager(load_denoiser=False)
sentences_manager = SentencesManager(os.path.join(cfg.SAMPLES_DIR, 'sentences.txt'),
                                     os.path.join(cfg.SAMPLES_DIR, 'samples/used_sentences.json'))
meta_object = load_meta_file()
embs_cache = load_embs_cache()

app = Flask(__name__, root_path=os.path.join(os.getcwd(), 'web_app'))


@app.route('/')
def home():
    filtered_name = request.cookies.get('identity_name')

    if filtered_name is None:
        return render_template('login.html', bad_name=request.args.get('bad_name'))
    else:
        num_samples = 0
        for name in meta_object:
            num_samples += len(meta_object[name]['refs']) + len(meta_object[name]['evls'])

        all_identities = None
        if filtered_name == 'pasqualecelella' or filtered_name == 'antonioroberto':
            all_identities = {}
            for fname in sorted(meta_object):
                oname = meta_object[fname]['name']
                all_identities[oname] = {}
                all_identities[oname]['refs'] = len(meta_object[fname]['refs'])
                all_identities[oname]['evls'] = len(meta_object[fname]['evls'])

        return render_template(
            'home.html',
            identity_name=meta_object[filtered_name]['name'],
            num_refs_by_user=len(meta_object[filtered_name]['refs']),
            num_evls_by_user=len(meta_object[filtered_name]['evls']),
            num_identities=len(meta_object),
            num_samples=num_samples,
            already_there=request.args.get('already_there'),
            all_identities=all_identities
        )


@app.route('/enroll')
def enroll():
    filtered_name = request.cookies.get('identity_name')
    if filtered_name is None:
        return make_response(redirect(url_for('home')))
    sentence_id = sentences_manager.get_next_senteces()[0]
    sentences_manager.mark_as_temp_used(sentence_id)
    sentence = sentences_manager.get_sentence_by_index(sentence_id)
    return render_template('enroll.html',
                           identity_name=meta_object[filtered_name]['name'],
                           sentence=sentence,
                           sentence_id=sentence_id)


@app.route('/test')
def test():
    filtered_name = request.cookies.get('identity_name')
    if filtered_name is None or len(meta_object[filtered_name]['refs']) < 3:
        return make_response(redirect(url_for('home')))
    sentence_id = sentences_manager.get_next_senteces()[0]
    sentences_manager.mark_as_temp_used(sentence_id)
    sentence = sentences_manager.get_sentence_by_index(sentence_id)
    return render_template('test.html',
                           identity_name=meta_object[filtered_name]['name'],
                           sentence=sentence,
                           sentence_id=sentence_id)


@app.route('/sentence')
def sentence():
    sentence_id = sentences_manager.get_next_senteces()[0]
    sentences_manager.mark_as_temp_used(sentence_id)
    sentence = sentences_manager.get_sentence_by_index(sentence_id)
    
    resp = dict()
    resp['sentence'] = sentence
    resp['sentence_id'] = sentence_id
    
    return jsonify(resp)


@app.route('/upload', methods = ['POST'])
def upload():
    filtered_name = request.cookies.get('identity_name')

    f = request.files['audio_data']
    operation = request.form['type_of_operation']  # enrollment or test ?
    file_id = request.form['file_id']
    sentence_id = int(request.form['sentence_id'])
    logger.info("Sentence ID: %d", sentence_id)

    if operation == 'enrollment':
        personal_folder = get_refs_path(filtered_name)
        filename = str(len(get_refs(filtered_name))).zfill(4)+'.wav'
    else:
        personal_folder = get_evls_path(filtered_name)
        filename = str(len(get_evls(filtered_name))).zfill(4)+'.wav'
    personal_folder = personal_folder.replace('refs', 'refs_before_conversion')
    personal_folder = personal_folder.replace('evls', 'evls_before_conversion')

    if not os.path.exists(personal_folder):
        os.makedirs(personal_folder)

    full_path = os.path.join(personal_folder, filename)
    with open(full_path, 'wb') as audio:
        f.save(audio)

    logging.info('%s uploaded a sample (%s), saved to: %s',
                 meta_object[filtered_name]['name'],
                 operation,
                 full_path)

    #######################
    # Conversion: wav/16000Hz/16bit
    full_path_after_conversion = full_path.replace('_before_conversion', '')
    convert_to_wav(full_path, full_path_after_conversion)

    # Duration
    max_duration = 3
    duration = len(audio_manager.load_audio_segment(full_path_after_conversion))/1000
    if duration < max_duration:
        resp = dict()
        resp['file_id'] = file_id
        resp['message'] = 'Duration too low (%.2fs). Please record a sample of greater duration.' % (duration)
        resp['status'] = 'error'
        os.remove(full_path)
        os.remove(full_path_after_conversion)
        return jsonify(resp)

    # SNR estimation
    SNR = audio_manager.compute_snr(filepaths=[full_path_after_conversion])[0]
    if operation == 'enrollment':
        if SNR < -5:
            resp = dict()
            resp['file_id'] = file_id
            resp['message'] = 'SNR (%ddB) is too low. Make sure you are in a quiet room and keep the microphone close enough.' % (SNR)
            resp['status'] = 'error'
            os.remove(full_path)
            os.remove(full_path_after_conversion)
            return jsonify(resp)

    # Embedding
    embedding = compute_embedding(audio_manager, full_path_after_conversion)
    embs_cache[get_short_path(full_path_after_conversion)] = embedding
    save_embs_cache(embs_cache)

    #######################
    # Update meta file
    if operation == 'enrollment':
        meta_object[filtered_name]['refs'].append(filename)
    else:
        meta_object[filtered_name]['evls'].append(filename)
    save_meta_file(meta_object)
    
    #######################
    # Update sentence manager
    sentences_manager.unmark_as_temp_used(sentence_id)
    sentences_manager.mark_as_used(sentence_id)
    
    #######################
    # Response
    resp = dict()
    resp['file_id'] = file_id
    if operation == 'enrollment':
        resp['message'] = '<strong>Success!</strong><br><strong>Estimated SNR:</strong> %ddB' % (SNR)
    else:
        top3_scores, top3_predictions, T = recognize(meta_object, embs_cache, embedding)
        prediction = meta_object[top3_predictions[0]]['name'] if top3_scores[0] >= T else 'Unknown'
        top3_formatted = ''
        for i in range(len(top3_scores)):
            full_name = meta_object[top3_predictions[i]]['name']
            top3_formatted += '<li>%s (score: %.2f)</li>' % (full_name, top3_scores[i])
        resp['message']  = '<strong>Success!</strong><br>'
        resp['message'] += '<strong>Estimated SNR:</strong> %ddB<br>' % (SNR)
        resp['message'] += '<strong>Prediction:</strong> %s<br>' % (prediction)
        resp['message'] += '<strong>Top3:</strong> <ul>%s</ul>' % (top3_formatted)
        logger.info('Score: %.2f / Prediction: %s / Threshold: %.2f / Top3: %s', top3_scores[0]*100, prediction, T*100, top3_predictions)
    resp['status'] = 'success'
    return jsonify(resp)


@app.route('/login', methods = ['POST'])
def login():
    name = request.form['name']
    if len(name) < 3:
        return make_response(redirect(url_for('home', bad_name=True)))

    session_id = request.cookies.get('session_id')
    if session_id is None:
        session_id = name+'_'+str(time.time())
        session_id = sha256(session_id.encode()).hexdigest()

    already_there = False
    filtered_name = re.sub('[^A-Za-z0-9]+', '', name).lower()
    if filtered_name not in meta_object:
        meta_object[filtered_name] = {}
        meta_object[filtered_name]['name'] = name
        meta_object[filtered_name]['refs'] = []
        meta_object[filtered_name]['evls'] = []
        meta_object[filtered_name]['session_id'] = session_id
        save_meta_file(meta_object)
    else:
        if 'session_id' in meta_object[filtered_name] and meta_object[filtered_name]['session_id'] != session_id:
            already_there = True

    if already_there:
        resp = make_response(redirect(url_for('home', already_there=already_there)))
    else:
        resp = make_response(redirect(url_for('home')))

    resp.set_cookie('identity_name', filtered_name, max_age=60*60*24*90)
    resp.set_cookie('session_id', session_id, max_age=60*60*24*90)

    return resp


@app.route('/logout')
def logout():
    resp = make_response(redirect('/'))

    filtered_name = request.cookies.get('identity_name')
    if filtered_name is not None:
        resp.set_cookie('identity_name', '', expires=0)

    return resp


@app.after_request
def add_header(r):
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r


if __name__ == '__main__':
    if os.path.exists('server.crt') and os.path.exists('server.key'):
        context = ('server.crt', 'server.key')
        app.run(host='0.0.0.0', ssl_context=context)
    else:
        app.run(host='0.0.0.0', ssl_context='adhoc')
    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
