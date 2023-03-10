import { platformBrowserDynamic } from '@angular/platform-browser-dynamic';

import { AppModule } from './app/app.module';
import { registerLicense } from '@syncfusion/ej2-base';

// Registering Syncfusion license key
registerLicense('Mgo+DSMBaFt/QHRqVVhjVFpFdEBBXHxAd1p/VWJYdVt5flBPcDwsT3RfQF5jSH1SdEVmW3xYeXNcRw==;Mgo+DSMBPh8sVXJ0S0J+XE9HflRDX3xKf0x/TGpQb19xflBPallYVBYiSV9jS31Td0ZiW39bcndSRmdZVQ==;ORg4AjUWIQA/Gnt2VVhkQlFadVdJXGFWfVJpTGpQdk5xdV9DaVZUTWY/P1ZhSXxQdkRgWH1edXZWQWFbV00=;MTA0MTkzMUAzMjMwMmUzNDJlMzBJeFAxczluckZ2Z0kwTTFQQkZTRnNWM3E3bmdkaDlkQk9pQm5VdWZvandZPQ==;MTA0MTkzMkAzMjMwMmUzNDJlMzBWSDV1SzA1cGZzTCtNR1A0ZmlHNGZvYjk2dUR1ZWQ1d09OdFZQNDRyc2JnPQ==;NRAiBiAaIQQuGjN/V0Z+WE9EaFxKVmJLYVB3WmpQdldgdVRMZVVbQX9PIiBoS35RdUViWn5ccHFXRWdfVUV+;MTA0MTkzNEAzMjMwMmUzNDJlMzBNdXF2V2I4TmlhdmJYSmovZ29CcjZXaEdQY0FpZEdYMDB4SWFBK01LUVZNPQ==;MTA0MTkzNUAzMjMwMmUzNDJlMzBMdjlYbFNpN0t3aFpwSVFhZHRsR2tRTVRpR0RSRFpIeFlkTlJPWXlmaVpnPQ==;Mgo+DSMBMAY9C3t2VVhkQlFadVdJXGFWfVJpTGpQdk5xdV9DaVZUTWY/P1ZhSXxQdkRgWH1edXZWQWZeVkE=;MTA0MTkzN0AzMjMwMmUzNDJlMzBPbVg5M28wdmxibDBKbUgxaUlhZU1mUy9YYXhqYmF6M2xMbVM3NVRTWEZBPQ==;MTA0MTkzOEAzMjMwMmUzNDJlMzBJV3l6SEJmby9Jd2x1TUNCaUorcTJUT3FPR2FLbXdrbmdlb1lHcENWMkgwPQ==;MTA0MTkzOUAzMjMwMmUzNDJlMzBNdXF2V2I4TmlhdmJYSmovZ29CcjZXaEdQY0FpZEdYMDB4SWFBK01LUVZNPQ==');

platformBrowserDynamic().bootstrapModule(AppModule)
  .catch(err => console.error(err));
