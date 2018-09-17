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

package xyz.safeflight.datacollection.PlotData;

import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.graphics.Color;
import android.support.v4.content.LocalBroadcastManager;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.WindowManager;
import android.widget.Toast;

import xyz.safeflight.datacollection.Constants;
import xyz.safeflight.datacollection.DataCollection.DataObject;
import xyz.safeflight.datacollection.R;
import com.github.mikephil.charting.charts.LineChart;
import com.github.mikephil.charting.components.Legend;
import com.github.mikephil.charting.components.XAxis;
import com.github.mikephil.charting.components.YAxis;
import com.github.mikephil.charting.data.Entry;
import com.github.mikephil.charting.data.LineData;
import com.github.mikephil.charting.data.LineDataSet;
import com.github.mikephil.charting.highlight.Highlight;
import com.github.mikephil.charting.interfaces.datasets.ILineDataSet;
import com.github.mikephil.charting.listener.OnChartValueSelectedListener;
import com.github.mikephil.charting.utils.ColorTemplate;

import java.util.ArrayList;

public class PlotActivity extends AppCompatActivity implements
        OnChartValueSelectedListener {
    private LineChart mChart;
    private String TAG = "PlotActivity";
    private BroadcastReceiver receiver;
    private LocalBroadcastManager sender;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        // Full screen? Not sure just checking it
        getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN,
                WindowManager.LayoutParams.FLAG_FULLSCREEN);
        setContentView(R.layout.activity_plot);


        // Register broadcast sender and receiver
        sender = LocalBroadcastManager.getInstance(this);
        // Register receiver to check location messages
        receiver = new BroadcastReceiver(){
            @Override
            public void onReceive(Context context, Intent intent) {
                processMessage(intent);
            }
        };
        // Add broadcast message filters
        addBdFilters();


        mChart = (LineChart) findViewById(R.id.chart1);

        mChart = (LineChart) findViewById(R.id.chart1);
        mChart.setOnChartValueSelectedListener(this);
        mChart.setDrawGridBackground(false);
        mChart.getDescription().setEnabled(false);

        // add an empty data object
        mChart.setData(new LineData());
//        mChart.getXAxis().setDrawLabels(false);
//        mChart.getXAxis().setDrawGridLines(false);

        mChart.invalidate();
        Log.e(TAG, "onCreate finished . . ");
    }


    @Override
    protected void onStop() {
        super.onStop();

    }


    private void addBdFilters(){
        Log.e(TAG,"ADDING FILTERS PLOT ACTIVITY");
        IntentFilter myfilter = new IntentFilter();
        myfilter.addAction(Constants.PLOT_VALUE);
        // Register the receiver
        LocalBroadcastManager.getInstance(this).registerReceiver((receiver),myfilter);
    }

    private void processMessage(Intent intent) {
        String action = intent.getAction();
        switch (action) {
            case Constants.PLOT_VALUE:
                Log.e(TAG, "New DataObject from LocationService");
                DataObject data = intent.getParcelableExtra(Constants.MESSAGE);
                addEntry(data);
                break;
        }
    }


    int[] mColors = ColorTemplate.VORDIPLOM_COLORS;

    private void addEntry(DataObject obj) {
        Log.e(TAG, "Adding new entry");
        LineData data = mChart.getData();

        ILineDataSet set = data.getDataSetByIndex(0);
        // set.addEntry(...); // can be called as well

        if (set == null) {
            set = createSet();
            data.addDataSet(set);
        }

        // choose a random dataSet
        int randomDataSetIndex = (int) (Math.random() * data.getDataSetCount());
        //float yValue = (float) (Math.random() * 10) + 50f;
        float yValue = (float) obj.getLocation().getLatitude();
        data.addEntry(new Entry(data.getDataSetByIndex(randomDataSetIndex).getEntryCount(), yValue), randomDataSetIndex);
        data.notifyDataChanged();

        // let the chart know it's data has changed
        mChart.notifyDataSetChanged();

        mChart.setVisibleXRangeMaximum(15);
        //mChart.setVisibleYRangeMaximum(15, AxisDependency.LEFT);
//
//            // this automatically refreshes the chart (calls invalidate())
        mChart.moveViewTo(data.getEntryCount() - 7, 50f, YAxis.AxisDependency.LEFT);

    }

    private void removeLastEntry() {
        Log.e(TAG, "Removing last entry...");
        LineData data = mChart.getData();

        if (data != null) {

            ILineDataSet set = data.getDataSetByIndex(0);

            if (set != null) {

                Entry e = set.getEntryForXValue(set.getEntryCount() - 1, Float.NaN);

                data.removeEntry(e, 0);
                // or remove by index
                // mData.removeEntryByXValue(xIndex, dataSetIndex);
                data.notifyDataChanged();
                mChart.notifyDataSetChanged();
                mChart.invalidate();
            }
        }
    }

    private void addDataSet() {
        Log.e(TAG, "Adding dataset");
        LineData data = mChart.getData();

        if (data != null) {

            int count = (data.getDataSetCount() + 1);

            ArrayList<Entry> yVals = new ArrayList<Entry>();

            for (int i = 0; i < data.getEntryCount(); i++) {
                yVals.add(new Entry(i, (float) (Math.random() * 50f) + 50f * count));
            }

            LineDataSet set = new LineDataSet(yVals, "DataSet " + count);
            set.setLineWidth(2.5f);
            set.setCircleRadius(4.5f);

            int color = mColors[count % mColors.length];

            set.setColor(color);
            set.setCircleColor(color);
            set.setHighLightColor(color);
            set.setValueTextSize(10f);
            set.setValueTextColor(color);

            data.addDataSet(set);
            data.notifyDataChanged();
            mChart.notifyDataSetChanged();
            mChart.invalidate();
        }
    }

    private void removeDataSet() {
        Log.e(TAG, "Removing dataset .  .");
        LineData data = mChart.getData();

        if (data != null) {

            data.removeDataSet(data.getDataSetByIndex(data.getDataSetCount() - 1));

            mChart.notifyDataSetChanged();
            mChart.invalidate();
        }
    }

    @Override
    public void onValueSelected(Entry e, Highlight h) {
        Toast.makeText(this, e.toString(), Toast.LENGTH_SHORT).show();
    }

    @Override
    public void onNothingSelected() {

    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        getMenuInflater().inflate(R.menu.plot_menu, menu);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {

        switch (item.getItemId()) {
            case R.id.actionAddEntry:
                //addEntry();
                Toast.makeText(this, "Entry added!", Toast.LENGTH_SHORT).show();
                break;
            case R.id.actionRemoveEntry:
                removeLastEntry();
                Toast.makeText(this, "Entry removed!", Toast.LENGTH_SHORT).show();
                break;
            case R.id.actionAddDataSet:
                addDataSet();
                Toast.makeText(this, "DataSet added!", Toast.LENGTH_SHORT).show();
                break;
            case R.id.actionRemoveDataSet:
                removeDataSet();
                Toast.makeText(this, "DataSet removed!", Toast.LENGTH_SHORT).show();
                break;
            case R.id.actionAddEmptyLineData:
                mChart.setData(new LineData());
                mChart.invalidate();
                Toast.makeText(this, "Empty data added!", Toast.LENGTH_SHORT).show();
                break;
            case R.id.actionClear:
                mChart.clear();
                Toast.makeText(this, "Chart cleared!", Toast.LENGTH_SHORT).show();
                break;
        }

        return true;
    }

    private LineDataSet createSet() {
        Log.e(TAG, "Creating dataset. . . ");
        LineDataSet set = new LineDataSet(null, "DataSet 1");
        set.setLineWidth(2.5f);
        set.setCircleRadius(4.5f);
        set.setColor(Color.rgb(240, 99, 99));
        set.setCircleColor(Color.rgb(240, 99, 99));
        set.setHighLightColor(Color.rgb(190, 190, 190));
        set.setAxisDependency(YAxis.AxisDependency.LEFT);
        set.setValueTextSize(10f);

        return set;
    }
}
