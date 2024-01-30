# SQL Queries practice

After log in to the PostgreSql server, study the example database called site_log. This is database implementation of a (site log)[https://www.freepik.com/premium-vector/construction-daily-site-log-book-kdp-interior_36692714.htm]

Define the following SQL queries.

List the names of all contractors

    select name from site_log.contractor

List the names and contract numbers of all contractors

    select name, contract_no from site_log.contractor

List the names and contract numbers of all contractors and rename contract_no to 'contract'

    select name, contract_no as contract from site_log.contractor

List the names and contract numbers of all contractors and rename contract_no to 'Contractor Number'

    select name, contract_no as "Contractor Number" from site_log.contractor

List all contractors and all attributes

    select * from site_log.contractor

List the names of all foremen

    select * from site_log.foreman

List the first and second name of all foremen

    select first_name, last_name from site_log.foreman

List foreman whose name is Joe

    select first_name, last_name from site_log.foreman where first_name = 'Joe' 

List foreman whose contractor number is 202212-A

    select * from site_log.foreman where contract_no = '202212-A' 

List those log entries which where afer Sept 15, 2022

    select * from site_log.log where enter > '2022-09-15'

List those log entries which where afer Sept 15, 2022 but berfore Sept 25, 2022

    select * from site_log.log where enter > '2022-09-15' AND enter < '2022-09-25' 

List those log entries which where afer Sept 15, 2022 0:00 am but before Sept 15, 2022 11 am

    select * from site_log.log where enter > '2022-09-13 00:00:00' AND enter < '2022-09-13 11:00:00'  

List those log entries which where on Sept 15, 2022.

    select * from site_log.log where enter > '2022-09-13 00:00:00' AND enter < '2022-09-14 00:00:00'  

List contractors and their trade

    select name, type from site_log.contractor, site_log.trade where trade_id = site_log.contractor.id

    select c.name, t.type from site_log.contractor as c, site_log.trade as t where c.trade_id = t.id
