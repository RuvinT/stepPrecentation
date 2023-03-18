import { Component } from '@angular/core';
import { Router } from '@angular/router';

@Component({
  selector: 'app-login',
  templateUrl: './login.component.html',
  styleUrls: ['./login.component.css']
})
export class LoginComponent {
  username: string;
  password: string;

  constructor(private router: Router) {}

  login() {
    if (this.username === 'Premier' && this.password === '1234' || this.username === 'Silver' && this.password === '1234') {
      let premior = true;
      if(this.username === 'Silver'){
        premior = false;
      }
      
    // Login successful, navigate to home page with premior value
    this.router.navigate(['/profile'], { queryParams: { premior: premior } });
    } else {
      // Login failed, display error message
      alert('Invalid username or password');
    }
  }
}

