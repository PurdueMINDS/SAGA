/*
 *  Copyright 2018 Jianfei Gao, Leonardo Teixeira, Bruno Ribeiro.
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 */

package xyz.safeflight.datacollection.DataCollection;

import android.app.NotificationManager;
import android.app.PendingIntent;
import android.content.Context;
import android.content.Intent;
import android.support.v4.app.TaskStackBuilder;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.support.v7.app.NotificationCompat;
import android.util.Log;
import android.view.View;
import android.widget.TextView;

import xyz.safeflight.datacollection.FileManagement.FileSingleton;
import xyz.safeflight.datacollection.R;
import xyz.safeflight.datacollection.Utils;

public class FrequencyActivity extends AppCompatActivity {
    private final String TAG = "Frequency Activity";
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_frequency);


    }

    public void handleStartButton(View v){
        final TextView text = (TextView) findViewById(R.id.editText);
        int period = Integer.parseInt(text.getText().toString());

        // Send user selected period to new activity
        Intent intent = new Intent(this, CollectData.class);
        intent.putExtra("period",period);

        // Create new file name = time
        FileSingleton file = FileSingleton.getInstance();
        file.setFilename(Utils.getTime("HH_mm_ss"));

        Log.e(TAG,"Filename is: " + file.getDirectory() + file.getFilename());

        startActivity(intent);
    }


}
