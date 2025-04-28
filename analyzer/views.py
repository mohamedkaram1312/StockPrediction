from django.shortcuts import render
from django.shortcuts import render
from .forms import StockAnalysisForm
from .stock_utils import analyze_stock

def index(request):
    form = StockAnalysisForm()
    result = None
    error = None
    warning = None
    
    if request.method == 'POST':
        form = StockAnalysisForm(request.POST)
        if form.is_valid():
            ticker = form.cleaned_data['ticker']
            start_date = form.cleaned_data['start_date']
            end_date = form.cleaned_data['end_date']
            
            # Convert dates to string format for yfinance
            start_date_str = start_date.strftime('%Y-%m-%d')
            end_date_str = end_date.strftime('%Y-%m-%d') if end_date else None
            
            # Call the analyze_stock function
            analysis_result = analyze_stock(ticker, start_date_str, end_date_str)
            
            if isinstance(analysis_result, dict) and 'error' in analysis_result:
                error = analysis_result['error']
            elif isinstance(analysis_result, dict) and 'warning' in analysis_result:
                warning = analysis_result['warning']
                if analysis_result.get('continue'):
                    result = analysis_result
            else:
                result = analysis_result
    
    context = {
        'form': form,
        'result': result,
        'error': error,
        'warning': warning
    }
    
    return render(request, 'analyzer/index.html', context)
# Create your views here.
