IO.popen ("date") { |f| puts f.gets }
system("mkdir apps")
system("cd apps")
system("rails new test_app")
