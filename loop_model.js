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

const MODEL_FILE_URL = 'loop_model/tensorflowjs_model.pb';
const WEIGHT_MANIFEST_FILE_URL = 'loop_model/weights_manifest.json';
const JSON_MODEL_FILE_URL = 'json_loop_model/model.json';
const OUTPUT_NODE_NAME = 'Add';

class LoopModel {
  constructor() {}

  async load() {
    this.model =
        await tf.loadGraphModel(MODEL_FILE_URL);
    this.jsonModel = await tf.loadGraphModel(JSON_MODEL_FILE_URL);
  }

  dispose() {
    if (this.model) {
      this.model.dispose();
    }
    if (this.jsonModel) {
      this.jsonModel.dispose();
    }
  }

  async predict(init, loop, loop2, inc, jsonModel) {
    const dict = {
      'init': tf.scalar(init, 'int32'),
      'times': tf.scalar(loop, 'int32'),
      'times2': tf.scalar(loop2, 'int32'),
      'inc': tf.scalar(inc, 'int32')
    };
    return jsonModel ? this.model.executeAsync(dict, OUTPUT_NODE_NAME) :
                       this.jsonModel.executeAsync(dict, OUTPUT_NODE_NAME);
  }
}

window.onload = async () => {
  const resultElement = document.getElementById('result');

  resultElement.innerText = 'Loading Control Flow model...';

  const loopModel = new LoopModel();
  console.time('Loading of model');
  await loopModel.load();
  console.timeEnd('Loading of model');
  resultElement.innerText = 'Model loaded.';

  const runBtn = document.getElementById('run');
  runBtn.onclick = async () => {
    const init = parseInt(document.getElementById('init').value);
    const loop = parseInt(document.getElementById('loop').value);
    const loop2 = parseInt(document.getElementById('loop2').value);
    const inc = parseInt(document.getElementById('inc').value);
    console.time('prediction');
    const result = await loopModel.predict(init, loop, loop2, inc);
    const jsonResult = await loopModel.predict(init, loop, loop2, inc, true);
    console.timeEnd('prediction');

    resultElement.innerText = 'pb output = ' + result.dataSync()[0];
    resultElement.innerText += ' json output = ' + jsonResult.dataSync()[0];
  };
};
