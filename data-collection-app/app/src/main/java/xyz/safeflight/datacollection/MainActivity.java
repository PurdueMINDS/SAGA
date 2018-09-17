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

package xyz.safeflight.datacollection;

import android.content.Intent;
import android.preference.PreferenceManager;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;

import xyz.safeflight.datacollection.DataCollection.FrequencyActivity;
import xyz.safeflight.datacollection.DataCollection.Sensors.SensorList;
import xyz.safeflight.datacollection.FileManagement.RootDirectoryList;
import xyz.safeflight.datacollection.FileManagement.FileSingleton;
import xyz.safeflight.datacollection.PlotData.PlotActivity;

public class MainActivity extends AppCompatActivity {
    private final String TAG = "MainActivity";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        final Button button = (Button) findViewById(R.id.collect);
        button.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                // Perform action on click
                collectData(v);
            }
        });

        // Get default setttings
        PreferenceManager.setDefaultValues(this, R.xml.preferences, false);
        // Set working directory for files with todays date
        FileSingleton file = FileSingleton.getInstance();
        file.setDirectory(Utils.getDate());
        Log.e(TAG, "MAIN ACTIVITYYYYYYYYYYYYY READY");
    }


    /** Called when the user clicks the Collect button */
    public void collectData (View view) {
        Intent intent = new Intent(this, FrequencyActivity.class);
        startActivity(intent);
    }

    public void checkFile (View view){
        Intent intent = new Intent(this, RootDirectoryList.class);
        startActivity(intent);
    }

    /*public void deleteFile (View view){
        boolean result = this.deleteFile(Constants.FILENAME);
        if (result) {
            Toast.makeText(getApplicationContext(), "File deleted succesfully !", Toast.LENGTH_LONG).show();
        } else {
            Toast.makeText(getApplicationContext(), "File not found !", Toast.LENGTH_LONG).show();
        }
    }*/

    public void showSensorList(View view){
        Intent intent = new Intent(this, SensorList.class);
        startActivity(intent);
    }

}
