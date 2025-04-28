from django import forms
from datetime import datetime

class StockAnalysisForm(forms.Form):
    ticker = forms.CharField(
        label="Stock Ticker",
        max_length=10,
        required=True,
        widget=forms.TextInput(attrs={'placeholder': 'E.g., AAPL, MSFT'})
    )
    
    start_date = forms.DateField(
        label="Start Date",
        required=True,
        widget=forms.DateInput(attrs={'type': 'date'}),
        initial='2020-01-01'
    )
    
    end_date = forms.DateField(
        label="End Date",
        required=False,
        widget=forms.DateInput(attrs={'type': 'date'}),
        initial=datetime.today().strftime('%Y-%m-%d')
    )
    
    def clean_ticker(self):
        ticker = self.cleaned_data.get('ticker')
        if ticker:
            return ticker.strip().upper()
        return ticker