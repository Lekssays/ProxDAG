module github.com/Lekssays/ProxDAG/network/committee

go 1.16

require (
	github.com/ProtonMail/go-ecvrf v0.0.1
	github.com/coreos/bbolt v1.3.2 // indirect
	github.com/dchest/blake2b v1.0.0 // indirect
	github.com/drand/drand v1.3.1 // indirect
	github.com/go-redis/redis/v8 v8.11.4
	github.com/golang/protobuf v1.5.2
	github.com/nikkolasg/slog v0.0.0-20170921200349-3c8d441d7a1e // indirect
	github.com/stretchr/testify v1.7.0 // indirect
	golang.org/x/net v0.0.0-20210813160813-60bc85c4be6d // indirect
	golang.org/x/sys v0.0.0-20210816183151-1e6c022a8912 // indirect
	golang.org/x/text v0.3.7 // indirect
	google.golang.org/protobuf v1.27.1
	gopkg.in/yaml.v3 v3.0.0-20210107192922-496545a6307b // indirect
)

replace (
	github.com/Lekssays/ProxDAG/network/committee/proto/peer => ./proto/peer
	github.com/Lekssays/ProxDAG/network/proxdag v0.0.0-20220225162616-d7378533521a => ../proxdag
)
