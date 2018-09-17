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

package xyz.safeflight.datacollection.DataCollection.Sensors;

import android.content.Context;
import android.hardware.Sensor;
import android.hardware.SensorManager;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.widget.ListView;
import android.widget.SimpleAdapter;

import xyz.safeflight.datacollection.R;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class SensorList extends AppCompatActivity {
    private ListView listView;
    private SensorManager mSensorManager;
    private List<Sensor> deviceSensors = null;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_sensor_list);

        listView = ((ListView) findViewById(R.id.listView));

        mSensorManager = (SensorManager) getSystemService(Context.SENSOR_SERVICE);
        deviceSensors = mSensorManager.getSensorList(Sensor.TYPE_ALL);
        List<Map<String,String>> data = new ArrayList<Map<String,String>>();

        for (Sensor sensor: deviceSensors){
            Map<String, String> datum = new HashMap<String, String>(2);
            datum.put("title", sensor.getName());
            datum.put("vendor", sensor.getVendor());
            data.add(datum);
        }
        SimpleAdapter adapter = new SimpleAdapter(this, data,
                android.R.layout.simple_list_item_2,
                new String[] {"title", "vendor"},
                new int[] {android.R.id.text1,
                        android.R.id.text2});

        listView.setAdapter(adapter);

    }
}
