import { Component, OnInit } from '@angular/core';
import { Chart, registerables } from 'chart.js';
import { ITooltipRenderEventArgs, IStockChartEventArgs,ChartTheme, IAxisLabelRenderEventArgs } from '@syncfusion/ej2-angular-charts';
import { saveAs } from 'file-saver';

declare var require: any;
var CanvasJS = require('../assets/canvasjs.min');
import { StockDataService } from './stock-data-service.service';
import {StockData} from './StockDataModel';

Chart.register(...registerables);
@Component({
    selector: 'app-root',
    templateUrl: './app.component.html',
    styleUrls: ['./app.component.css']
})
export class AppComponent {
   
    title = 'Apple';
    public data1: Object[];

    constructor(private stockDataService: StockDataService) {
        this.stockDataService.getStockData().subscribe(data => {
          
            const stockDataArray: StockData[] = data.map((item: StockData) => {
                return {
                  close: item.close,
                  high: item.high,
                  low: item.low,
                  open: item.open,
                  volume: item.volume,
                  x: item.x
                };
              });    
          this.data1 = stockDataArray;
        });
        
    }
    /**
     * Type of series to be drawn in radar or polar series. They are
     *  'Line'
     *  'Column'
     *  'Area'
     *  'Scatter'
     *  'Spline'
     *  'StackingColumn'
     *  'StackingArea'
     *  'RangeColumn'
     *  'SplineArea'
     *
     * @default 'Line'
     */
    public seriesType: string[] = ['Line','OHLC','Spline','Candle'];

    public indicatorType: string[] = ['Macd'];
    public trendlineType: string[]=['Linear','Exponential','Polynomial','Moving Average'] ;  
    
    public primaryXAxis: Object = {
        valueType: 'DateTime', majorGridLines: { width: 0 }, crosshairTooltip: { enable: true }
    };

    public primaryYAxis: Object = {
        lineStyle: { color: 'transparent' },
        majorTickLines: { color: 'transparent', width: 0 }
    };
    public chartArea: Object = {
        border: {
            width: 0
        }
    };
    public crosshair: Object = {
        enable: true
    };
    public tooltip: object = { enable: true };
    public columnTooltip: boolean = false;
    public tooltipRender(args: ITooltipRenderEventArgs): void {
        if ((args.text || '').split('<br/>')[4]) {
            let target: number = parseInt((args.text || '').split('<br/>')[4].split('<b>')[1].split('</b>')[0], 10);
            let value: string = (target / 100000000).toFixed(1) + 'B';
            args.text = (args.text || '').replace((args.text || '').split('<br/>')[4].split('<b>')[1].split('</b>')[0], value);
        }
    };
    public axisLabelRender(args: IAxisLabelRenderEventArgs): void {
        let text: number = parseInt(args.text, 10);
        if (args.axis.name === 'primaryYAxis') {
            args.text = text / 100000000 + 'B';
        }
    };
    public load(args: IStockChartEventArgs): void {
        let selectedTheme: string = location.hash.split('/')[1];
        selectedTheme = selectedTheme ? selectedTheme : 'Material';
        args.stockChart.theme = <ChartTheme>(selectedTheme.charAt(0).toUpperCase() + selectedTheme.slice(1)).replace(/-dark/i, "Dark");
    };
    


    //Second chart
    dataPoints1: any[] = [];
    dataPoints2: any[] = [];

    chart: any;

    chartOptions = {
        zoomEnabled: true,
        theme: "light1",
        title: {
            text: "Share value of Apple and Windows"
        },
        axisX: {
            title: "chart updates every 2 secs"
        },
        axisY: {
            prefix: "$"
        },
        toolTip: {
            shared: true
        },
        legend: {
            cursor: "pointer",
            verticalAlign: "top",
            fontSize: 22,
            fontColor: "dimGrey",
            itemclick: function (e: any) {
                if (typeof (e.dataSeries.visible) === "undefined" || e.dataSeries.visible) {
                    e.dataSeries.visible = false;
                }
                else {
                    e.dataSeries.visible = true;
                }
                e.chart.render();
            }
        },
        data: [{
            type: "line",
            xValueType: "dateTime",
            yValueFormatString: "$####.00",
            xValueFormatString: "hh:mm:ss TT",
            showInLegend: true,
            name: "Apple",
            dataPoints: this.dataPoints1
        }, {
            type: "line",
            xValueType: "dateTime",
            yValueFormatString: "$####.00",
            showInLegend: true,
            name: "Microsoft",
            dataPoints: this.dataPoints2
        }]
    }

    getChartInstance(chart: object) {
        this.chart = chart;

        this.time.setHours(9);
        this.time.setMinutes(30);
        this.time.setSeconds(0);
        this.time.setMilliseconds(0);
        this.updateChart(100);
    }

    updateInterval = 2000;

    // initial value
    yValue1 = 90;
    yValue2 = 97;
    time = new Date();

    updateChart = (count: any) => {
        count = count || 1;
        var deltaY1, deltaY2;
        for (var i = 0; i < count; i++) {
            this.time.setTime(this.time.getTime() + this.updateInterval);
            deltaY1 = .5 - Math.random();
            deltaY2 = .5 - Math.random();

            // adding random value and rounding it to two digits. 
            this.yValue1 = Math.round((this.yValue1 + deltaY1) * 100) / 100;
            this.yValue2 = Math.round((this.yValue2 + deltaY2) * 100) / 100;

            // pushing the new values
            this.dataPoints1.push({
                x: this.time.getTime(),
                y: this.yValue1
            });
            this.dataPoints2.push({
                x: this.time.getTime(),
                y: this.yValue2
            });
        }

        // updating legend text with  updated with y Value 
        this.chart.options.data[0].legendText = " Apple  $" + CanvasJS.formatNumber(this.yValue1, "#,###.00");
        this.chart.options.data[1].legendText = " Microsoft  $" + CanvasJS.formatNumber(this.yValue2, "#,###.00");
        this.chart.render();
    }
    ngAfterViewInit() {
        setInterval(() => {
            this.updateChart(1);
        }, this.updateInterval);
        
        //let jsonData = JSON.stringify(this.data1);
        //const file = new Blob([jsonData], { type: 'text/plain;charset=utf-8' });
        //saveAs(file, 'stock-data.txt');

        //console.log('Data saved to file');
    }

}