select 
	"Marital Status",
	avg(age) as Mean
	from customer
	group by "Marital Status";

select
	case 
		when gender=0 then 'Wanita'
		when gender=1 then 'Pria'
		else '-'
		end as gender,
	avg(age) as Mean
	from customer 
	group by gender;

select 
	storename,
	SUM(qty)as total_quantity
	from store 
	join transaction on store.storeid = transaction.storeid 
	group by storename
	order by total_quantity desc
	limit 1;

select 
	product."Product Name",
	SUM(TotalAmount)as total_amount
	from product 
	join transaction on product.productID = transaction.productID 
	group by product."Product Name"
	order by total_amount desc
	limit 1;
	
	
	



	
	

