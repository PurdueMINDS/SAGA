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
