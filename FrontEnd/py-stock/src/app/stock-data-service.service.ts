import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class StockDataService {

  constructor(private http: HttpClient) {}

  getStockData(): Observable<any> {
    return this.http.get('http://127.0.0.1:5000/stock_prices');
  }
}
