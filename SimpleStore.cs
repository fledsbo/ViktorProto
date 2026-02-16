namespace ViktorProto
{
    using global::Viktor;
    using LiteDB;
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using System.Text;
    using System.Threading.Tasks;
    using static System.Runtime.InteropServices.JavaScript.JSType;

    internal class SimpleStore : IStore
    {
        private readonly LiteDatabase db = new LiteDatabase(@"Filename=MyData.db; Mode=Exclusive");
        private readonly ILiteCollection<BsonDocument> items;

        public SimpleStore()
        {
            items = db.GetCollection<BsonDocument>("items");
            items.EnsureIndex("_id");
        }

        public void UpsertItem(int key, byte[] data)
        {
            var doc = new BsonDocument
            {
                ["_id"] = key,
                ["data"] = data // Store byte[] directly; LiteDB BsonDocument supports byte[] for binary data
            };
            items.Upsert(doc);
        }

        public bool ReadItem(int key, out byte[] data)
        {
            var doc = items.FindById(key);
            if (doc != null)
            {
                data = doc["data"].AsBinary;
                return true;
            }
            data = null;
            return false;
        }

        public IEnumerable<int> ReadAllItems()
        {
            return items.FindAll().Select(doc => doc["_id"].AsInt32);
        }

        public IEnumerable<int> FindItems(string query)
        {
            throw new NotImplementedException();
        }
    }
}
