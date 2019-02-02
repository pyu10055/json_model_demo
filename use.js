/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
const BASE_PATH =
    'https://storage.googleapis.com/tfjs-models/savedmodel/universal_sentence_encoder/';

const flatten = (arr) =>
    arr.reduce((acc, curr) => acc.concat(curr), []);

const loadModel = async () => {
    return await tf.loadFrozenModel(
        `${BASE_PATH}tensorflowjs_model.pb`,
        `${BASE_PATH}weights_manifest.json`);
  }

  const loadVocabulary = async () => {
    const vocabulary = await fetch(`${BASE_PATH}vocab.json`);
    return await vocabulary.json();
  }

const load = async () => {
    const [model, vocabulary] =
        await Promise.all([loadModel(), loadVocabulary()]);
    document.querySelector("#load-status").textContent = "loading complete";
  const encodings = [[184, 147, 1341]];
    const indicesArr =
        flatten(encodings.map((arr, i) => arr.map((d, index) => [i, index])));
    const indices =
        tf.tensor2d(flatten(indicesArr), [indicesArr.length, 2], 'int32');
    const values = tf.tensor1d(flatten(encodings), 'int32');
  document.querySelector("#before").textContent = tf.memory().numTensors;
 
    const embeddings = await model.executeAsync({indices, values});
    const embeddings1 = await model.executeAsync({indices, values});
    const embeddings2 = await model.executeAsync({indices, values});
    console.log(embeddings);
    indices.dispose();
    values.dispose();
  document.querySelector("#after").textContent = tf.memory().numTensors;
}

load();